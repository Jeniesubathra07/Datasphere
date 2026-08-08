"""Load and validate the Stage 2 ML artifact at application startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasphere.ml.artifacts import load_metadata, load_pipeline
from datasphere.ml.config import MODEL_ARTIFACT_NAME, METADATA_ARTIFACT_NAME


class ModelLoadError(RuntimeError):
    """Raised when the production model cannot be loaded."""


@dataclass
class LoadedModel:
    pipeline: Any
    metadata: dict[str, Any]
    model_path: Path
    metadata_path: Path


def load_production_model(models_dir: Path) -> LoadedModel:
    model_path = models_dir / MODEL_ARTIFACT_NAME
    metadata_path = models_dir / METADATA_ARTIFACT_NAME

    if not model_path.exists():
        raise ModelLoadError(
            f"Model artifact not found at {model_path}. "
            "Run: python scripts/run_stage2_pipeline.py"
        )
    if not metadata_path.exists():
        raise ModelLoadError(f"Metadata artifact not found at {metadata_path}")

    pipeline = load_pipeline(models_dir)
    metadata = load_metadata(models_dir)

    if not metadata.get("feature_schema"):
        raise ModelLoadError("Model metadata is missing feature_schema")

    if not hasattr(pipeline, "predict"):
        raise ModelLoadError("Loaded artifact does not expose predict()")

    return LoadedModel(
        pipeline=pipeline,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )
