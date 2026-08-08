"""Model explainability — global and local."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasphere.ml.config import RISK_LEVELS


def _get_feature_names(pipeline, sample: pd.DataFrame) -> list[str]:
    engineered = pipeline.named_steps["feature_engineer"].transform(sample)
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transform(engineered).shape[1])]


def global_feature_importance(
    pipeline,
    x_sample: pd.DataFrame,
    *,
    top_n: int = 15,
) -> list[dict[str, float]]:
    """Return global feature importance from tree-based models or coefficients."""
    classifier = pipeline.named_steps["classifier"]
    engineered = pipeline.named_steps["feature_engineer"].transform(x_sample)
    feature_names = _get_feature_names(pipeline, x_sample)

    importances: np.ndarray | None = None
    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)
    elif hasattr(classifier, "coef_"):
        importances = np.mean(np.abs(classifier.coef_), axis=0)

    if importances is None or len(importances) != len(feature_names):
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": name, "importance": float(val)} for name, val in pairs[:top_n]]


def explain_prediction(
    pipeline,
    student_row: pd.DataFrame,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Local explanation for a single student record."""
    if len(student_row) != 1:
        raise ValueError("student_row must contain exactly one record")

    predicted = pipeline.predict(student_row)[0]
    explanation: dict[str, Any] = {
        "predicted_risk": str(predicted),
        "class_probabilities": {},
        "important_features": [],
        "human_readable_explanation": "",
    }

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(student_row)[0]
        classes = list(pipeline.named_steps["classifier"].classes_)
        explanation["class_probabilities"] = {
            str(cls): float(p) for cls, p in zip(classes, proba)
        }

    global_imp = global_feature_importance(pipeline, student_row, top_n=top_n)
    explanation["important_features"] = global_imp

    top_feats = ", ".join(item["feature"] for item in global_imp[:3])
    explanation["human_readable_explanation"] = (
        f"The model predicts '{predicted}' risk. "
        f"Among the strongest global contributors in this model are: {top_feats}. "
        "This reflects statistical association in training data, not proven causation."
    )
    return explanation


def error_analysis(
    pipeline,
    x_val: pd.DataFrame,
    y_true: pd.Series,
) -> dict[str, Any]:
    """Analyze misclassifications on validation data."""
    predictions = pipeline.predict(x_val)
    y_true_arr = y_true.astype(str).to_numpy()
    pred_arr = np.asarray(predictions, dtype=str)

    misclassified_mask = y_true_arr != pred_arr
    misclassified = int(misclassified_mask.sum())
    total = len(y_true_arr)

    confusion_pairs: dict[str, int] = {}
    for actual, pred in zip(y_true_arr[misclassified_mask], pred_arr[misclassified_mask]):
        key = f"{actual} -> {pred}"
        confusion_pairs[key] = confusion_pairs.get(key, 0) + 1

    minority_errors = {
        level: int(((y_true_arr == level) & misclassified_mask).sum())
        for level in RISK_LEVELS
        if level in set(y_true_arr)
    }

    return {
        "misclassification_rate": misclassified / total if total else 0.0,
        "misclassified_count": misclassified,
        "total_evaluated": total,
        "top_confusion_pairs": sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10],
        "minority_class_errors": minority_errors,
        "limitations": [
            "Adjacent risk levels (e.g. Medium vs High) may be inherently ambiguous.",
            "Minority classes (Critical, High) have fewer training examples.",
            "Model uses association patterns — not causal intervention effects.",
        ],
    }
