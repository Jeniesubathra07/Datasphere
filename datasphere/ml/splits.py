"""Reproducible train / validation / test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from datasphere.ml.config import RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE


@dataclass
class DataSplits:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    strategy: str
    random_seed: int


def create_splits(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
) -> DataSplits:
    """
    Stratified train / validation / test split.

    Strategy: hold out a final test set first (20%), then split the remaining
    training pool into train (80%) and validation (20%) — yielding ~64/16/20 overall.
    Stratification preserves class proportions because dropout risk is imbalanced
    and there is no single temporal prediction cutoff in Stage 1.
    """
    stratify = target if target.nunique() > 1 else None

    x_temp, x_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    relative_val_size = validation_size
    stratify_temp = y_temp if y_temp.nunique() > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=stratify_temp,
    )

    return DataSplits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        strategy="stratified_holdout_64_16_20",
        random_seed=random_seed,
    )
