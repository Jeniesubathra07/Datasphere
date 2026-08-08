"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from datasphere.api.schemas.prediction import HealthResponse
from datasphere.core.config import APP_VERSION

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "loaded_model", None)
    return HealthResponse(
        status="ok",
        platform="EduRisk Intelligence",
        model_loaded=loaded is not None,
        model_version=(
            str(loaded.metadata.get("model_version")) if loaded is not None else None
        ),
    )
