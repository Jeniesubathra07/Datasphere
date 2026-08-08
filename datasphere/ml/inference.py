"""Production inference API for the saved pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datasphere.data.paths import MODELS_DIR
from datasphere.ml.artifacts import load_metadata, load_pipeline
from datasphere.ml.cleaning import clean_dataset
from datasphere.ml.config import EXCLUDED_FEATURE_COLUMNS, TARGET_COLUMN
from datasphere.ml.explain import explain_prediction
from datasphere.ml.features import split_features_target
from datasphere.ml.validation import REQUIRED_COLUMNS


# Inference accepts the full raw student schema; target is optional at prediction time.
REQUIRED_INPUT_COLUMNS = tuple(col for col in REQUIRED_COLUMNS if col != TARGET_COLUMN)


class InferenceError(ValueError):
    """Raised when inference input fails validation."""


def _prepare_inference_frame(student: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and strip excluded columns for model input."""
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in student.columns]
    if missing:
        raise InferenceError(f"Missing required input columns: {missing}")

    extra = set(student.columns) - set(REQUIRED_COLUMNS)
    if extra:
        raise InferenceError(f"Unexpected columns in input: {sorted(extra)}")

    cleaned = clean_dataset(student)
    features, _ = split_features_target(cleaned)
    return features


def predict_student_risk(
    student: pd.DataFrame | dict[str, Any],
    *,
    models_dir: Path | None = None,
    explain: bool = True,
) -> dict[str, Any]:
    """
    Run inference on one student record using the saved production pipeline.

    Accepts the same raw schema the future API will use.
    """
    models_dir = models_dir or MODELS_DIR
    pipeline = load_pipeline(models_dir)

    if isinstance(student, dict):
        frame = pd.DataFrame([student])
    else:
        frame = student.copy()

    model_input = _prepare_inference_frame(frame)

    predicted = pipeline.predict(model_input)[0]
    result: dict[str, Any] = {"predicted_risk": str(predicted)}

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(model_input)[0]
        classes = list(pipeline.named_steps["classifier"].classes_)
        result["class_probabilities"] = {str(c): float(p) for c, p in zip(classes, proba)}

    if explain:
        result["explanation"] = explain_prediction(pipeline, model_input)

    return result


def build_synthetic_validation_student(raw_template: pd.DataFrame) -> dict[str, Any]:
    """Create one unseen row for technical inference validation (not a hardcoded prediction)."""
    if raw_template.empty:
        raise ValueError("Template dataframe is empty")
    row = raw_template.iloc[[-1]].copy()
    return row.iloc[0].to_dict()
