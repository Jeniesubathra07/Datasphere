"""Pydantic schemas for prediction responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskPrediction(BaseModel):
    risk_level: str
    probabilities: dict[str, float]


class ExplanationFactor(BaseModel):
    feature: str
    importance: float


class PredictionExplanation(BaseModel):
    important_factors: list[ExplanationFactor] = Field(default_factory=list)
    summary: str = ""


class ModelInfo(BaseModel):
    version: str
    selected_model: str
    primary_metric: str | None = None


class PredictionResponse(BaseModel):
    prediction: RiskPrediction
    explanation: PredictionExplanation | None = None
    model: ModelInfo


class HealthResponse(BaseModel):
    status: str
    platform: str
    model_loaded: bool
    model_version: str | None = None


class ErrorResponse(BaseModel):
    detail: str
