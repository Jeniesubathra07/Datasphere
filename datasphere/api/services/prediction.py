"""Prediction service — thin wrapper over Stage 2 ML pipeline."""

from __future__ import annotations

from typing import Any

from datasphere.api.schemas.prediction import (
    ExplanationFactor,
    ModelInfo,
    PredictionExplanation,
    PredictionResponse,
    RiskPrediction,
)
from datasphere.api.schemas.student import StudentPredictionRequest
from datasphere.core.model_loader import LoadedModel
from datasphere.ml.inference import InferenceError, predict_with_pipeline


def run_prediction(
    request: StudentPredictionRequest,
    loaded: LoadedModel,
    *,
    explain: bool = True,
) -> PredictionResponse:
    try:
        raw = predict_with_pipeline(
            loaded.pipeline,
            request.to_inference_dict(),
            explain=explain,
        )
    except InferenceError as exc:
        raise ValueError(str(exc)) from exc

    explanation = None
    if explain and "explanation" in raw:
        exp = raw["explanation"]
        explanation = PredictionExplanation(
            important_factors=[
                ExplanationFactor(feature=f["feature"], importance=f["importance"])
                for f in exp.get("important_features", [])
            ],
            summary=exp.get("human_readable_explanation", ""),
        )

    metadata = loaded.metadata
    return PredictionResponse(
        prediction=RiskPrediction(
            risk_level=raw["predicted_risk"],
            probabilities=raw.get("class_probabilities", {}),
        ),
        explanation=explanation,
        model=ModelInfo(
            version=str(metadata.get("model_version", "unknown")),
            selected_model=str(metadata.get("selected_model", "unknown")),
            primary_metric=metadata.get("primary_metric"),
        ),
    )
