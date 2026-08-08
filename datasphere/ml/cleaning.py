"""Cleaning transforms — always operate on a copy; raw data stays untouched."""

from __future__ import annotations

import pandas as pd

from datasphere.ml.config import CATEGORICAL_MISSING_FILL

NUMERIC_IMPUTE_COLUMNS: tuple[str, ...] = (
    "distance_to_school_km",
    "household_income_monthly_usd",
    "family_size",
    "attendance_rate_pct",
    "average_test_score_pct",
    "teacher_student_ratio",
    "school_infrastructure_score",
)

CATEGORICAL_IMPUTE_COLUMNS: tuple[str, ...] = (
    "mother_education_level",
    "father_education_level",
    "household_has_internet_access",
)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataset following Stage 1 imputation rules."""
    cleaned = df.copy()

    for col in NUMERIC_IMPUTE_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in CATEGORICAL_IMPUTE_COLUMNS:
        if col in cleaned.columns:
            if col == "household_has_internet_access":
                cleaned[col] = cleaned[col].astype(str).replace({"True": "Yes", "False": "No", "nan": CATEGORICAL_MISSING_FILL})
                cleaned[col] = cleaned[col].fillna(CATEGORICAL_MISSING_FILL)
            else:
                cleaned[col] = cleaned[col].astype(str)
                mode = cleaned[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else CATEGORICAL_MISSING_FILL
                cleaned[col] = cleaned[col].replace("nan", fill).fillna(fill)

    if "case_notes" in cleaned.columns:
        cleaned["case_notes"] = cleaned["case_notes"].fillna("No notes")

    return cleaned
