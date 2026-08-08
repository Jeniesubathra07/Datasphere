"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request

from datasphere.core.model_loader import LoadedModel


def get_loaded_model(request: Request) -> LoadedModel:
    loaded = getattr(request.app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the pipeline first: python scripts/run_stage2_pipeline.py",
        )
    return loaded
