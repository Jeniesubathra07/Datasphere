"""Model metadata and training routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from datasphere.api.deps import get_loaded_model
from datasphere.api.schema_fields import CATEGORICAL_OPTIONS, NUMERIC_BOUNDS, PREDICT_API_FIELDS
from datasphere.api.schemas.student import ModelSchemaResponse
from datasphere.core.model_loader import LoadedModel
from datasphere.data.loader import discover_datasets, load_primary_dataset
from datasphere.ml.config import EXCLUDED_FEATURE_COLUMNS, RISK_LEVELS
from datasphere.ml.train import train_model

router = APIRouter(prefix="/api", tags=["model"])


@router.get("/model-info", response_model=dict)
def model_info(loaded: LoadedModel = Depends(get_loaded_model)) -> dict:
    meta = loaded.metadata
    return {
        "version": meta.get("model_version"),
        "selected_model": meta.get("selected_model"),
        "primary_metric": meta.get("primary_metric"),
        "class_labels": meta.get("class_labels", list(RISK_LEVELS)),
        "test_metrics": meta.get("test_metrics"),
        "feature_schema": meta.get("feature_schema"),
        "excluded_features": meta.get("excluded_features"),
    }


@router.get("/model/schema", response_model=ModelSchemaResponse)
def model_schema(loaded: LoadedModel = Depends(get_loaded_model)) -> ModelSchemaResponse:
    return ModelSchemaResponse(
        fields=list(PREDICT_API_FIELDS),
        categorical_options=CATEGORICAL_OPTIONS,
        numeric_bounds={k: [v[0], v[1]] for k, v in NUMERIC_BOUNDS.items()},
        excluded_fields=list(EXCLUDED_FEATURE_COLUMNS),
        class_labels=loaded.metadata.get("class_labels", list(RISK_LEVELS)),
    )


@router.get("/datasets")
def list_datasets() -> dict[str, object]:
    datasets = discover_datasets()
    if not datasets:
        return {
            "datasets": [],
            "message": "No CSV files found. Add datasets under datasets/raw/.",
        }
    return {"datasets": datasets}


@router.post("/model/train")
def train_risk_model() -> dict[str, object]:
    try:
        frame = load_primary_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return train_model(frame)
