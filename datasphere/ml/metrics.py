"""Evaluation metrics for multiclass risk classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from datasphere.ml.config import PRIMARY_METRIC, RISK_LEVELS


def compute_metrics(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    label_order = labels or list(RISK_LEVELS)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=label_order)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=label_order)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=label_order)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0, labels=label_order)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=label_order).tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=label_order,
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics["primary_metric"] = PRIMARY_METRIC
    metrics["primary_metric_value"] = metrics[PRIMARY_METRIC]
    return metrics


def primary_score(metrics: dict[str, Any]) -> float:
    return float(metrics.get(PRIMARY_METRIC, metrics.get("primary_metric_value", 0.0)))
