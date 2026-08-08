"""Prediction routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from datasphere.api.deps import get_loaded_model
from datasphere.api.schemas.prediction import PredictionResponse
from datasphere.api.schemas.student import StudentPredictionRequest
from datasphere.api.services.prediction import run_prediction
from datasphere.core.model_loader import LoadedModel

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    body: StudentPredictionRequest,
    loaded: LoadedModel = Depends(get_loaded_model),
) -> PredictionResponse:
    try:
        return run_prediction(body, loaded, explain=False)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/predict/explain", response_model=PredictionResponse)
def predict_with_explanation(
    body: StudentPredictionRequest,
    loaded: LoadedModel = Depends(get_loaded_model),
) -> PredictionResponse:
    try:
        return run_prediction(body, loaded, explain=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
