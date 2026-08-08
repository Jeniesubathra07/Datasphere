"""Dataset validation — schema checks without modifying raw data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from datasphere.ml.config import EXCLUDED_FEATURE_COLUMNS, RISK_LEVELS, TARGET_COLUMN

REQUIRED_COLUMNS: tuple[str, ...] = (
    "student_id",
    "record_generated_date",
    "enrollment_date",
    "academic_year",
    "country",
    "state_province",
    "location_type",
    "age",
    "gender",
    "grade_level",
    "school_type",
    "distance_to_school_km",
    "mode_of_transport",
    "household_income_monthly_usd",
    "mother_education_level",
    "father_education_level",
    "family_size",
    "number_of_siblings",
    "is_orphan",
    "single_parent_household",
    "household_has_electricity",
    "household_has_internet_access",
    "owns_smartphone_or_computer",
    "child_labor_involvement",
    "attendance_rate_pct",
    "average_test_score_pct",
    "previous_year_pass",
    "number_of_school_transfers",
    "disciplinary_incidents_count",
    "bullying_incidents_reported",
    "health_issues_reported",
    "special_needs_status",
    "free_meal_program_enrolled",
    "scholarship_received",
    "extracurricular_participation",
    "teacher_student_ratio",
    "school_infrastructure_score",
    "early_marriage_risk_flag",
    "community_conflict_zone",
    "seasonal_migration_family",
    "ngo_intervention_present",
    "literacy_program_enrolled",
    "language_barrier_flag",
    "report_source",
    "case_notes",
    "dropout_probability_score",
    TARGET_COLUMN,
)


@dataclass
class ValidationReport:
    passed: bool
    row_count: int
    column_count: int
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    duplicate_student_ids: int = 0
    invalid_target_values: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def validate_raw_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate schema and basic integrity of the raw dataset."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    unexpected = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    duplicate_ids = int(df["student_id"].duplicated().sum()) if "student_id" in df.columns else 0

    invalid_targets: list[str] = []
    if TARGET_COLUMN in df.columns:
        values = set(df[TARGET_COLUMN].dropna().astype(str).unique())
        invalid_targets = sorted(values - set(RISK_LEVELS))

    messages: list[str] = []
    if missing:
        messages.append(f"Missing required columns: {missing}")
    if duplicate_ids:
        messages.append(f"Duplicate student_id count: {duplicate_ids}")
    if invalid_targets:
        messages.append(f"Invalid target values: {invalid_targets}")

    passed = not missing and not invalid_targets
    return ValidationReport(
        passed=passed,
        row_count=len(df),
        column_count=len(df.columns),
        missing_columns=missing,
        unexpected_columns=unexpected,
        duplicate_student_ids=duplicate_ids,
        invalid_target_values=invalid_targets,
        messages=messages,
    )


def assert_no_leakage_columns(feature_columns: list[str]) -> None:
    """Raise if any leakage or excluded column appears in the feature matrix."""
    forbidden = set(EXCLUDED_FEATURE_COLUMNS) - {TARGET_COLUMN}
    present = forbidden.intersection(feature_columns)
    if present:
        raise ValueError(f"Leakage or excluded columns found in X: {sorted(present)}")
