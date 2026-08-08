"""Stage 2 ML pipeline configuration — approved Stage 1 decisions."""

from __future__ import annotations

RANDOM_SEED = 42

TARGET_COLUMN = "dropout_risk_level"

# Columns that must never appear in X (identifiers, leakage, target, unapproved text).
EXCLUDED_FEATURE_COLUMNS: tuple[str, ...] = (
    "student_id",
    "dropout_probability_score",
    "dropout_risk_level",
    "case_notes",
)

# Raw date columns replaced by engineered features before modeling.
DATE_COLUMNS: tuple[str, ...] = (
    "record_generated_date",
    "enrollment_date",
)

ENGINEERED_DATE_FEATURES: tuple[str, ...] = (
    "record_year",
    "record_month",
    "enrollment_year",
    "enrollment_month",
    "enrollment_duration_days",
)

ENGINEERED_NUMERIC_FEATURES: tuple[str, ...] = (
    "household_income_log",
)

RISK_LEVELS: tuple[str, ...] = ("Low", "Medium", "High", "Critical")

# Primary model-selection metric (approved Stage 1 objective for imbalanced risk classes).
PRIMARY_METRIC = "macro_f1"

# Train / validation / test proportions (test is held out first and never used in tuning).
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # fraction of the post-test-split training pool

NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
CATEGORICAL_MISSING_FILL = "Unknown"

MODEL_ARTIFACT_NAME = "education_risk_pipeline.joblib"
METADATA_ARTIFACT_NAME = "education_risk_pipeline_metadata.json"
