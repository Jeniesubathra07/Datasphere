"""Model benchmarking — baseline and candidate models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from datasphere.ml.config import PRIMARY_METRIC, RANDOM_SEED
from datasphere.ml.features import build_model_pipeline
from datasphere.ml.metrics import compute_metrics, primary_score


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    params: dict[str, Any]
    train_time_seconds: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    validation_metrics: dict[str, Any]
    notes: str = ""


@dataclass
class BenchmarkReport:
    baseline: ModelResult
    candidates: list[ModelResult] = field(default_factory=list)
    imbalance_strategy: str = ""
    imbalance_notes: str = ""


def _evaluate_pipeline(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    cv_folds: int = 3,
) -> tuple[float, float, dict[str, Any], float]:
    folds = 2 if len(x_train) > 80_000 else cv_folds
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    val_pred = pipeline.predict(x_val)
    val_metrics = compute_metrics(y_val, val_pred, labels=sorted(y_train.unique()))
    return float(cv_scores.mean()), float(cv_scores.std()), val_metrics, train_time


def _candidate_specs(use_class_weight: bool) -> list[tuple[str, Any, dict[str, Any], str]]:
    cw = "balanced" if use_class_weight else None
    return [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_SEED,
                class_weight=cw,
            ),
            {"class_weight": cw},
            "Linear baseline with standardized numeric features",
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=12,
                random_state=RANDOM_SEED,
                class_weight=cw,
            ),
            {"max_depth": 12, "class_weight": cw},
            "Non-linear baseline, interpretable",
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=75,
                max_depth=16,
                random_state=RANDOM_SEED,
                class_weight=cw,
                n_jobs=-1,
            ),
            {"n_estimators": 75, "max_depth": 16, "class_weight": cw},
            "Bagged trees for non-linear interactions",
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                class_weight=cw,
            ),
            {"max_iter": 200, "learning_rate": 0.1, "class_weight": cw},
            "Gradient boosting — strong tabular candidate",
        ),
    ]


def run_benchmark(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    use_class_weight: bool = True,
) -> BenchmarkReport:
    """Train baseline and candidate models; record CV and validation metrics."""
    baseline_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    baseline_pipe = build_model_pipeline(baseline_clf, x_train)
    cv_mean, cv_std, val_metrics, train_time = _evaluate_pipeline(
        baseline_pipe, x_train, y_train, x_val, y_val, cv_folds=3
    )
    baseline = ModelResult(
        name="baseline_stratified_dummy",
        pipeline=baseline_pipe,
        params={"strategy": "stratified"},
        train_time_seconds=train_time,
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        validation_metrics=val_metrics,
        notes="Reference point — predicts class distribution only",
    )

    candidates: list[ModelResult] = []
    for name, estimator, params, notes in _candidate_specs(use_class_weight):
        pipe = build_model_pipeline(estimator, x_train)
        cv_mean, cv_std, val_metrics, train_time = _evaluate_pipeline(
            pipe, x_train, y_train, x_val, y_val, cv_folds=3
        )
        candidates.append(
            ModelResult(
                name=name,
                pipeline=pipe,
                params=params,
                train_time_seconds=train_time,
                cv_macro_f1_mean=cv_mean,
                cv_macro_f1_std=cv_std,
                validation_metrics=val_metrics,
                notes=notes,
            )
        )

    return BenchmarkReport(
        baseline=baseline,
        candidates=candidates,
        imbalance_strategy="class_weight=balanced" if use_class_weight else "none",
        imbalance_notes=(
            "Compared balanced class weights because Critical/High classes are minority. "
            "SMOTE was not applied — resampling can distort calibrated probabilities."
        ),
    )


def comparison_table(report: BenchmarkReport) -> list[dict[str, Any]]:
    """Build model comparison table for Phase 10."""
    rows: list[dict[str, Any]] = []

    def _row(result: ModelResult) -> dict[str, Any]:
        vm = result.validation_metrics
        return {
            "model": result.name,
            "cv_macro_f1_mean": round(result.cv_macro_f1_mean, 4),
            "cv_macro_f1_std": round(result.cv_macro_f1_std, 4),
            "validation_macro_f1": round(vm["macro_f1"], 4),
            "validation_weighted_f1": round(vm["weighted_f1"], 4),
            "validation_precision_macro": round(vm["macro_precision"], 4),
            "validation_recall_macro": round(vm["macro_recall"], 4),
            "training_time_seconds": round(result.train_time_seconds, 2),
            "notes": result.notes,
        }

    rows.append(_row(report.baseline))
    rows.extend(_row(c) for c in report.candidates)
    return rows


def select_best_model(report: BenchmarkReport) -> ModelResult:
    """Select final model by validation macro F1 (primary metric)."""
    all_models = [report.baseline, *report.candidates]
    return max(all_models[1:], key=lambda m: primary_score(m.validation_metrics))
