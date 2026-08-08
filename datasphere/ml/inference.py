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
from datasphere.ml.features import extract_features_for_prediction
from datasphere.ml.validation import REQUIRED_COLUMNS


# Full raw schema columns (includes fields excluded from the model).
REQUIRED_INPUT_COLUMNS = tuple(col for col in REQUIRED_COLUMNS if col != TARGET_COLUMN)

# API prediction schema — excludes leakage, target, identifiers, and free text.
PREDICT_INPUT_COLUMNS = tuple(
    col
    for col in REQUIRED_COLUMNS
    if col not in EXCLUDED_FEATURE_COLUMNS and col != TARGET_COLUMN
)


class InferenceError(ValueError):
    """Raised when inference input fails validation."""


def _prepare_inference_frame(student: pd.DataFrame, *, strict_raw_schema: bool = False) -> pd.DataFrame:
    """Validate, clean, and strip excluded columns for model input."""
    if strict_raw_schema:
        missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in student.columns]
        if missing:
            raise InferenceError(f"Missing required input columns: {missing}")
        extra = set(student.columns) - set(REQUIRED_COLUMNS)
        if extra:
            raise InferenceError(f"Unexpected columns in input: {sorted(extra)}")
    else:
        missing = [col for col in PREDICT_INPUT_COLUMNS if col not in student.columns]
        if missing:
            raise InferenceError(f"Missing required input columns: {missing}")
        forbidden = set(EXCLUDED_FEATURE_COLUMNS) | {TARGET_COLUMN}
        extra = set(student.columns) - set(PREDICT_INPUT_COLUMNS) - forbidden
        if extra:
            raise InferenceError(f"Unexpected columns in input: {sorted(extra)}")

    cleaned = clean_dataset(student)
    return extract_features_for_prediction(cleaned)


def predict_with_pipeline(
    pipeline,
    student: pd.DataFrame | dict[str, Any],
    *,
    explain: bool = True,
) -> dict[str, Any]:
    """Run inference using a pre-loaded pipeline (no disk reload)."""
    frame = pd.DataFrame([student]) if isinstance(student, dict) else student.copy()
    model_input = _prepare_inference_frame(frame, strict_raw_schema=False)

    predicted = pipeline.predict(model_input)[0]
    result: dict[str, Any] = {"predicted_risk": str(predicted)}

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(model_input)[0]
        classes = list(pipeline.named_steps["classifier"].classes_)
        result["class_probabilities"] = {str(c): float(p) for c, p in zip(classes, proba)}

    if explain:
        result["explanation"] = explain_prediction(pipeline, model_input)

    return result


def predict_student_risk(
    student: pd.DataFrame | dict[str, Any],
    *,
    models_dir: Path | None = None,
    explain: bool = True,
) -> dict[str, Any]:
    """Run inference on one student record using the saved production pipeline."""
    models_dir = models_dir or MODELS_DIR
    pipeline = load_pipeline(models_dir)
    return predict_with_pipeline(pipeline, student, explain=explain)


def build_synthetic_validation_student(raw_template: pd.DataFrame) -> dict[str, Any]:
    """Create one unseen row for technical inference validation (not a hardcoded prediction)."""
    if raw_template.empty:
        raise ValueError("Template dataframe is empty")
    row = raw_template.iloc[[-1]].copy()
    record = row.iloc[0].to_dict()
    for key in list(record):
        if key in EXCLUDED_FEATURE_COLUMNS or key == TARGET_COLUMN:
            record.pop(key, None)
    return {k: v for k, v in record.items() if k in PREDICT_INPUT_COLUMNS}
