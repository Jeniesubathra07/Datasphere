"""Stage 2 production ML pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from datasphere.data.loader import load_primary_dataset
from datasphere.data.paths import MODELS_DIR
from datasphere.ml.artifacts import load_pipeline, save_artifacts
from datasphere.ml.benchmark import (
    comparison_table,
    run_benchmark,
    select_best_model,
)
from datasphere.ml.cleaning import clean_dataset
from datasphere.ml.config import (
    EXCLUDED_FEATURE_COLUMNS,
    PRIMARY_METRIC,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from datasphere.ml.explain import error_analysis, explain_prediction, global_feature_importance
from datasphere.ml.feature_engineering import FeatureEngineer
from datasphere.ml.features import split_features_target
from datasphere.ml.inference import build_synthetic_validation_student, predict_student_risk
from datasphere.ml.inference import _prepare_inference_frame
from datasphere.ml.metrics import compute_metrics
from datasphere.ml.splits import create_splits
from datasphere.ml.tuning import select_best_tuned, tune_top_models
from datasphere.ml.validation import validate_raw_dataset


@dataclass
class PipelineReport:
    validation_report: Any
    x_shape: tuple[int, int]
    y_shape: tuple[int]
    feature_count: int
    target_distribution: dict[str, int]
    leakage_checks: dict[str, bool]
    feature_columns: list[str]
    excluded_columns: list[str]
    split_strategy: str
    imbalance_decision: dict[str, str]
    model_comparison: list[dict[str, Any]]
    selected_model: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    error_analysis: dict[str, Any]
    global_importance: list[dict[str, float]]
    inference_example: dict[str, Any]
    artifact_paths: dict[str, str]
    limitations: list[str] = field(default_factory=list)


def _print_target_separation(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    dist = target.value_counts().to_dict()
    info = {
        "x_shape": features.shape,
        "y_shape": (len(target),),
        "feature_count": features.shape[1],
        "target_distribution": dist,
    }
    print("X shape:", info["x_shape"])
    print("y shape:", info["y_shape"])
    print("Feature count:", info["feature_count"])
    print("Target distribution:", dist)
    return info


def _verify_leakage(features: pd.DataFrame) -> dict[str, bool]:
    forbidden = set(EXCLUDED_FEATURE_COLUMNS)
    checks = {
        "target_not_in_x": TARGET_COLUMN not in features.columns,
        "student_id_absent": "student_id" not in features.columns,
        "dropout_probability_score_absent": "dropout_probability_score" not in features.columns,
        "dropout_risk_level_absent": "dropout_risk_level" not in features.columns,
        "case_notes_absent": "case_notes" not in features.columns,
        "no_leakage_columns": not forbidden.intersection(features.columns) - {TARGET_COLUMN},
    }
    for name, ok in checks.items():
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {name}")
    return checks


def run_stage2_pipeline(
    df: pd.DataFrame | None = None,
    *,
    models_dir: Path | None = None,
    sample_rows: int | None = None,
    use_class_weight: bool = True,
    run_tuning: bool = True,
) -> PipelineReport:
    """
    Execute the full Stage 2 ML pipeline end-to-end.

    Raw data is never modified in place.
    """
    models_dir = models_dir or MODELS_DIR
    raw = load_primary_dataset() if df is None else df.copy()

    if sample_rows is not None and len(raw) > sample_rows:
        raw = raw.sample(n=sample_rows, random_state=RANDOM_SEED).reset_index(drop=True)

    print("=" * 60)
    print("PHASE 1 — VALIDATION & CLEANING")
    print("=" * 60)
    validation_report = validate_raw_dataset(raw)
    if not validation_report.passed:
        raise ValueError(f"Dataset validation failed: {validation_report.messages}")
    print(f"Validated {validation_report.row_count:,} rows, {validation_report.column_count} columns")

    cleaned = clean_dataset(raw)

    print("\n" + "=" * 60)
    print("PHASE 2 — TARGET SEPARATION")
    print("=" * 60)
    features, target = split_features_target(cleaned)
    separation = _print_target_separation(features, target)
    leakage_checks = _verify_leakage(features)
    if not all(leakage_checks.values()):
        raise ValueError("Leakage checks failed")

    excluded = [c for c in EXCLUDED_FEATURE_COLUMNS if c in cleaned.columns]

    print("\n" + "=" * 60)
    print("PHASE 3 — TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)
    splits = create_splits(features, target)
    print(f"Split strategy: {splits.strategy} (seed={splits.random_seed})")
    print(f"Train: {splits.x_train.shape}, Val: {splits.x_val.shape}, Test: {splits.x_test.shape}")

    print("\n" + "=" * 60)
    print("PHASE 5-7 — BASELINE, BENCHMARK, CLASS IMBALANCE")
    print("=" * 60)
    minority_pct = (target.isin(["High", "Critical"]).mean()) * 100
    print(f"Minority classes (High+Critical): {minority_pct:.1f}% of data")
    benchmark = run_benchmark(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        use_class_weight=use_class_weight,
    )
    print(f"Imbalance strategy: {benchmark.imbalance_strategy}")
    print(benchmark.imbalance_notes)
    comparison = comparison_table(benchmark)
    for row in comparison:
        print(row)

    print("\n" + "=" * 60)
    print("PHASE 9 — HYPERPARAMETER TUNING")
    print("=" * 60)
    best_bench = select_best_model(benchmark)
    tuned_results = []
    if run_tuning:
        top_for_tuning = sorted(
            benchmark.candidates,
            key=lambda m: m.validation_metrics["macro_f1"],
            reverse=True,
        )[:2]
        tuned_results = tune_top_models(
            top_for_tuning,
            splits.x_train,
            splits.y_train,
            splits.x_val,
            splits.y_val,
        )
        for t in tuned_results:
            print(f"Tuned {t.name}: CV={t.cv_best_score:.4f}, val macro F1={t.validation_metrics['macro_f1']:.4f}")

    selected_name, final_pipeline, val_metrics = select_best_tuned(best_bench, tuned_results)
    print(f"\nSelected model: {selected_name}")

    print("\n" + "=" * 60)
    print("PHASE 11 — FINAL TEST EVALUATION (untouched test set)")
    print("=" * 60)
    test_pred = final_pipeline.predict(splits.x_test)
    test_metrics = compute_metrics(splits.y_test, test_pred, labels=sorted(target.unique()))
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")
    print("Confusion matrix:", test_metrics["confusion_matrix"])

    print("\n" + "=" * 60)
    print("PHASE 12-13 — ERROR ANALYSIS & EXPLAINABILITY")
    print("=" * 60)
    err = error_analysis(final_pipeline, splits.x_val, splits.y_val)
    importance = global_feature_importance(final_pipeline, splits.x_val.head(500))
    print("Top errors:", err["top_confusion_pairs"][:5])

    print("\n" + "=" * 60)
    print("PHASE 14-15 — SAVE ARTIFACT & INFERENCE TEST")
    print("=" * 60)
    engineered_cols = FeatureEngineer().fit_transform(features).columns.tolist()
    model_path, metadata_path = save_artifacts(
        final_pipeline,
        metadata={
            "selected_model": selected_name,
            "feature_schema": engineered_cols,
            "excluded_features": excluded,
            "class_labels": sorted(target.unique()),
            "target_column": TARGET_COLUMN,
            "split_strategy": splits.strategy,
            "primary_metric": PRIMARY_METRIC,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_comparison": comparison,
            "imbalance_strategy": benchmark.imbalance_strategy,
            "tuning_params": [t.best_params for t in tuned_results],
            "leakage_checks": leakage_checks,
            "random_seed": RANDOM_SEED,
        },
        models_dir=models_dir,
    )

    # Reload and verify
    reloaded = load_pipeline(models_dir)
    synthetic_student = build_synthetic_validation_student(cleaned)
    inference_result = predict_student_risk(synthetic_student, models_dir=models_dir)

    # Reload artifact and confirm identical prediction
    reloaded = load_pipeline(models_dir)
    reloaded_input = _prepare_inference_frame(pd.DataFrame([synthetic_student]))
    reloaded_pred = reloaded.predict(reloaded_input)[0]
    assert str(reloaded_pred) == inference_result["predicted_risk"], "Reloaded model prediction mismatch"

    print("Inference example:", inference_result["predicted_risk"])
    print("Reload check: PASS")
    print("Artifact:", model_path)

    return PipelineReport(
        validation_report=validation_report,
        x_shape=separation["x_shape"],
        y_shape=separation["y_shape"],
        feature_count=separation["feature_count"],
        target_distribution=separation["target_distribution"],
        leakage_checks=leakage_checks,
        feature_columns=engineered_cols,
        excluded_columns=excluded,
        split_strategy=splits.strategy,
        imbalance_decision={
            "strategy": benchmark.imbalance_strategy,
            "notes": benchmark.imbalance_notes,
        },
        model_comparison=comparison,
        selected_model=selected_name,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        error_analysis=err,
        global_importance=importance,
        inference_example=inference_result,
        artifact_paths={
            "model": str(model_path),
            "metadata": str(metadata_path),
        },
        limitations=err.get("limitations", []),
    )
