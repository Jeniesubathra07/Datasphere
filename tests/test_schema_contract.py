"""Verify frontend/API/ML field contract consistency."""

from __future__ import annotations

from datasphere.api.schema_fields import PREDICT_API_FIELDS
from datasphere.api.schemas.student import StudentPredictionRequest
from datasphere.ml.inference import PREDICT_INPUT_COLUMNS


def test_api_schema_matches_ml_inference_columns() -> None:
    api_fields = set(StudentPredictionRequest.model_fields.keys())
    ml_fields = set(PREDICT_INPUT_COLUMNS)
    assert api_fields == ml_fields, f"Mismatch: api-only={api_fields - ml_fields}, ml-only={ml_fields - api_fields}"


def test_predict_api_fields_constant() -> None:
    assert set(PREDICT_API_FIELDS) == set(PREDICT_INPUT_COLUMNS)
