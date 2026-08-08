"""Stage 2 ML pipeline validation tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from datasphere.data.loader import load_primary_dataset
from datasphere.ml.artifacts import load_metadata, load_pipeline
from datasphere.ml.cleaning import clean_dataset
from datasphere.ml.config import EXCLUDED_FEATURE_COLUMNS, TARGET_COLUMN
from datasphere.ml.features import split_features_target
from datasphere.ml.inference import InferenceError, predict_student_risk
from datasphere.ml.pipeline import run_stage2_pipeline
from datasphere.ml.validation import validate_raw_dataset


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    df = load_primary_dataset()
    return df.sample(n=2000, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory, sample_df: pd.DataFrame):
    models_dir = tmp_path_factory.mktemp("models")
    report = run_stage2_pipeline(
        sample_df,
        models_dir=models_dir,
        run_tuning=False,
    )
    return models_dir, report


def test_target_absent_from_x(sample_df: pd.DataFrame) -> None:
    cleaned = clean_dataset(sample_df)
    x, y = split_features_target(cleaned)
    assert TARGET_COLUMN not in x.columns
    assert len(y) == len(x)


def test_leakage_columns_absent(sample_df: pd.DataFrame) -> None:
    cleaned = clean_dataset(sample_df)
    x, _ = split_features_target(cleaned)
    for col in ("student_id", "dropout_probability_score", "dropout_risk_level", "case_notes"):
        assert col not in x.columns


def test_student_id_absent(sample_df: pd.DataFrame) -> None:
    cleaned = clean_dataset(sample_df)
    x, _ = split_features_target(cleaned)
    assert "student_id" not in x.columns


def test_validation_passes(sample_df: pd.DataFrame) -> None:
    report = validate_raw_dataset(sample_df)
    assert report.passed


def test_pipeline_trains(trained_artifacts) -> None:
    _, report = trained_artifacts
    assert report.feature_count > 0
    assert all(report.leakage_checks.values())
    assert report.test_metrics["macro_f1"] > 0


def test_artifact_saves_and_loads(trained_artifacts) -> None:
    models_dir, _ = trained_artifacts
    pipeline = load_pipeline(models_dir)
    metadata = load_metadata(models_dir)
    assert pipeline is not None
    assert metadata["target_column"] == TARGET_COLUMN
    assert "student_id" in metadata["excluded_features"]


def test_model_predicts(trained_artifacts, sample_df: pd.DataFrame) -> None:
    models_dir, _ = trained_artifacts
    pipeline = load_pipeline(models_dir)
    cleaned = clean_dataset(sample_df.head(5))
    x, _ = split_features_target(cleaned)
    preds = pipeline.predict(x)
    assert len(preds) == 5


def test_probability_output(trained_artifacts, sample_df: pd.DataFrame) -> None:
    models_dir, _ = trained_artifacts
    row = clean_dataset(sample_df.tail(1))
    result = predict_student_risk(row.iloc[0].to_dict(), models_dir=models_dir)
    assert "predicted_risk" in result
    assert "class_probabilities" in result
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 0.01


def test_class_labels_correct(trained_artifacts) -> None:
    _, report = trained_artifacts
    labels = set(report.target_distribution.keys())
    assert labels.issubset({"Low", "Medium", "High", "Critical"})


def test_missing_inputs_rejected(trained_artifacts, sample_df: pd.DataFrame) -> None:
    models_dir, _ = trained_artifacts
    bad = {"age": 12, "gender": "M"}
    with pytest.raises(InferenceError):
        predict_student_risk(bad, models_dir=models_dir)


def test_invalid_extra_column_rejected(trained_artifacts, sample_df: pd.DataFrame) -> None:
    models_dir, _ = trained_artifacts
    row = clean_dataset(sample_df.tail(1)).iloc[0].to_dict()
    row["unexpected_column"] = "bad"
    with pytest.raises(InferenceError):
        predict_student_risk(row, models_dir=models_dir)


def test_reloaded_artifact_same_prediction(trained_artifacts, sample_df: pd.DataFrame) -> None:
    models_dir, _ = trained_artifacts
    row = clean_dataset(sample_df.tail(1)).iloc[0].to_dict()
    r1 = predict_student_risk(row, models_dir=models_dir)
    # Force reload from disk
    pipeline = joblib.load(models_dir / "education_risk_pipeline.joblib")
    from datasphere.ml.inference import _prepare_inference_frame

    prepared = _prepare_inference_frame(pd.DataFrame([row]))
    r2 = pipeline.predict(prepared)[0]
    assert r1["predicted_risk"] == str(r2)
