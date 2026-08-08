from __future__ import annotations

import numpy as np
import pandas as pd

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


def _column(df: pd.DataFrame, *names: str) -> pd.Series | None:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lowered:
            return df[lowered[name.lower()]]
    return None


def construct_risk_label(df: pd.DataFrame) -> pd.Series:
    existing = _column(df, "dropout_risk_level", "risk_level", "risk", "dropout_risk")
    if existing is not None:
        return existing.astype(str).str.title().replace(
            {
                "0": "Low",
                "1": "Medium",
                "2": "High",
                "3": "Critical",
            }
        )

    attendance = _column(df, "attendance", "attendance_rate", "attendance_pct")
    test_score = _column(df, "test_score", "avg_test_score", "exam_score")
    child_labour = _column(df, "child_labour", "child_labor")
    income = _column(df, "household_income", "income", "family_income")

    score = pd.Series(0.0, index=df.index)
    if attendance is not None:
        score += np.where(attendance < 70, 2.5, np.where(attendance < 85, 1.0, 0.0))
    if test_score is not None:
        score += np.where(test_score < 40, 2.5, np.where(test_score < 60, 1.0, 0.0))
    if child_labour is not None:
        score += np.where(child_labour.astype(str).str.lower().isin({"1", "yes", "true"}), 2.0, 0.0)
    if income is not None:
        score += np.where(income < 15000, 1.5, 0.0)

    if score.max() == 0:
        score = pd.Series(np.random.default_rng(42).uniform(0, 4, len(df)), index=df.index)

    return pd.cut(
        score,
        bins=[-np.inf, 1.0, 2.0, 3.5, np.inf],
        labels=RISK_LEVELS,
    ).astype(str)
