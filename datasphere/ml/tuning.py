"""Hyperparameter tuning for top candidate models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from datasphere.ml.benchmark import ModelResult
from datasphere.ml.config import RANDOM_SEED
from datasphere.ml.features import build_model_pipeline
from datasphere.ml.metrics import compute_metrics, primary_score


@dataclass
class TuningResult:
    name: str
    best_pipeline: Any
    best_params: dict[str, Any]
    cv_best_score: float
    validation_metrics: dict[str, Any]
    tuning_time_seconds: float


def tune_top_models(
    top_results: list[ModelResult],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_iter: int = 8,
) -> list[TuningResult]:
    """Tune strongest candidates using CV on training data only."""
    tuned: list[TuningResult] = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    param_spaces: dict[str, dict[str, list[Any]]] = {
        "random_forest": {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [12, 16, 20, None],
            "classifier__min_samples_leaf": [1, 2, 5],
        },
        "hist_gradient_boosting": {
            "classifier__max_iter": [150, 200, 300],
            "classifier__learning_rate": [0.05, 0.1, 0.15],
            "classifier__max_depth": [6, 8, 10],
            "classifier__min_samples_leaf": [10, 20, 40],
        },
    }

    for result in top_results:
        if result.name not in param_spaces:
            continue

        base_estimator = result.pipeline.named_steps["classifier"]
        search_pipeline = build_model_pipeline(base_estimator, x_train)

        start = time.perf_counter()
        search = RandomizedSearchCV(
            search_pipeline,
            param_distributions=param_spaces[result.name],
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        tuning_time = time.perf_counter() - start

        val_pred = search.best_estimator_.predict(x_val)
        val_metrics = compute_metrics(y_val, val_pred, labels=sorted(y_train.unique()))

        tuned.append(
            TuningResult(
                name=result.name,
                best_pipeline=search.best_estimator_,
                best_params=search.best_params_,
                cv_best_score=float(search.best_score_),
                validation_metrics=val_metrics,
                tuning_time_seconds=tuning_time,
            )
        )

    return tuned


def select_best_tuned(
    benchmark_best: ModelResult,
    tuned_results: list[TuningResult],
) -> tuple[str, Any, dict[str, Any]]:
    """Pick best between untuned benchmark winner and tuned variants."""
    best_name = benchmark_best.name
    best_pipeline = benchmark_best.pipeline
    best_metrics = benchmark_best.validation_metrics

    for tuned in tuned_results:
        if primary_score(tuned.validation_metrics) > primary_score(best_metrics):
            best_name = f"{tuned.name}_tuned"
            best_pipeline = tuned.best_pipeline
            best_metrics = tuned.validation_metrics

    return best_name, best_pipeline, best_metrics
