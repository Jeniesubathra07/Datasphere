"""Backward-compatible training entry point — delegates to Stage 2 pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasphere.data.paths import MODELS_DIR
from datasphere.ml.labels import construct_risk_label
from datasphere.ml.pipeline import run_stage2_pipeline


def prepare_labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    if "dropout_risk_level" not in labeled.columns:
        labeled["dropout_risk_level"] = construct_risk_label(labeled)
    return labeled


def train_model(df: pd.DataFrame, model_path: Path | None = None) -> dict[str, object]:
    """Train the production pipeline and return summary metrics."""
    report = run_stage2_pipeline(df, models_dir=model_path.parent if model_path else MODELS_DIR)
    return {
        "model_path": report.artifact_paths["model"],
        "metadata_path": report.artifact_paths["metadata"],
        "rows": report.x_shape[0],
        "features": report.feature_count,
        "selected_model": report.selected_model,
        "classification_report": report.test_metrics["classification_report"],
        "confusion_matrix": report.test_metrics["confusion_matrix"],
        "labels": list(report.target_distribution.keys()),
        "test_metrics": report.test_metrics,
        "validation_metrics": report.validation_metrics,
    }
