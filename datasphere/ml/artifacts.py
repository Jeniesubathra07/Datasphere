"""Artifact persistence and metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from datasphere.ml.config import METADATA_ARTIFACT_NAME, MODEL_ARTIFACT_NAME, RANDOM_SEED, RISK_LEVELS


def save_artifacts(
    pipeline,
    metadata: dict[str, Any],
    models_dir: Path,
) -> tuple[Path, Path]:
    """Save complete inference pipeline and JSON metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_ARTIFACT_NAME
    metadata_path = models_dir / METADATA_ARTIFACT_NAME

    joblib.dump(pipeline, model_path)

    metadata_payload = {
        "model_version": metadata.get("model_version", "2.0.0"),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": metadata.get("random_seed", RANDOM_SEED),
        "selected_model": metadata.get("selected_model"),
        "feature_schema": metadata.get("feature_schema", []),
        "excluded_features": metadata.get("excluded_features", []),
        "class_labels": metadata.get("class_labels", list(RISK_LEVELS)),
        "target_column": metadata.get("target_column"),
        "split_strategy": metadata.get("split_strategy"),
        "primary_metric": metadata.get("primary_metric"),
        "validation_metrics": metadata.get("validation_metrics"),
        "test_metrics": metadata.get("test_metrics"),
        "model_comparison": metadata.get("model_comparison"),
        "imbalance_strategy": metadata.get("imbalance_strategy"),
        "tuning_params": metadata.get("tuning_params"),
        "leakage_checks": metadata.get("leakage_checks"),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_pipeline(models_dir: Path):
    """Load saved inference pipeline."""
    model_path = models_dir / MODEL_ARTIFACT_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def load_metadata(models_dir: Path) -> dict[str, Any]:
    metadata_path = models_dir / METADATA_ARTIFACT_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))
