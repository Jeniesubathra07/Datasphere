"""Approved feature engineering — sklearn-compatible transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from datasphere.ml.config import DATE_COLUMNS, ENGINEERED_DATE_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create approved prediction-time features from cleaned data."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        if "record_generated_date" in frame.columns:
            record_dt = pd.to_datetime(frame["record_generated_date"], errors="coerce")
            frame["record_year"] = record_dt.dt.year
            frame["record_month"] = record_dt.dt.month

        if "enrollment_date" in frame.columns:
            enrollment_dt = pd.to_datetime(frame["enrollment_date"], errors="coerce")
            frame["enrollment_year"] = enrollment_dt.dt.year
            frame["enrollment_month"] = enrollment_dt.dt.month

        if {"record_generated_date", "enrollment_date"}.issubset(frame.columns):
            record_dt = pd.to_datetime(frame["record_generated_date"], errors="coerce")
            enrollment_dt = pd.to_datetime(frame["enrollment_date"], errors="coerce")
            frame["enrollment_duration_days"] = (record_dt - enrollment_dt).dt.days

        if "household_income_monthly_usd" in frame.columns:
            income = pd.to_numeric(frame["household_income_monthly_usd"], errors="coerce")
            frame["household_income_log"] = np.log1p(income.clip(lower=0))

        for col in DATE_COLUMNS:
            frame = frame.drop(columns=[col], errors="ignore")

        for col in ENGINEERED_DATE_FEATURES:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        return frame
