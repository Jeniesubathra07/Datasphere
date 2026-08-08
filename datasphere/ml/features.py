"""Feature/target separation and sklearn preprocessing."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from datasphere.ml.config import (
    CATEGORICAL_IMPUTE_STRATEGY,
    CATEGORICAL_MISSING_FILL,
    EXCLUDED_FEATURE_COLUMNS,
    NUMERIC_IMPUTE_STRATEGY,
    TARGET_COLUMN,
)
from datasphere.ml.feature_engineering import FeatureEngineer
from datasphere.ml.validation import assert_no_leakage_columns


def _bool_to_float(array):
    return array.astype(float)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if pd.api.types.is_bool_dtype(frame[col]):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _boolean_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if pd.api.types.is_bool_dtype(frame[col])]


def _categorical_columns(frame: pd.DataFrame, numeric: list[str], boolean: list[str]) -> list[str]:
    excluded = set(numeric) | set(boolean)
    return [col for col in frame.columns if col not in excluded]


def extract_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Return model features without requiring the target column."""
    drop_cols = [col for col in EXCLUDED_FEATURE_COLUMNS if col in df.columns]
    features = df.drop(columns=drop_cols, errors="ignore")
    assert_no_leakage_columns(features.columns.tolist())
    return features


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Separate approved predictors (X) and target (y)."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    drop_cols = [col for col in EXCLUDED_FEATURE_COLUMNS if col in df.columns]
    features = df.drop(columns=drop_cols, errors="ignore")
    target = df[target_col].astype(str)

    assert_no_leakage_columns(features.columns.tolist())
    if target_col in features.columns:
        raise ValueError(f"Target column '{target_col}' must not appear in X")

    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn preprocessing for numeric, categorical, and boolean columns."""
    numeric_cols = _numeric_columns(features)
    boolean_cols = _boolean_columns(features)
    categorical_cols = _categorical_columns(features, numeric_cols, boolean_cols)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY, fill_value=CATEGORICAL_MISSING_FILL)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    bool_pipeline = Pipeline(
        steps=[
            (
                "to_int",
                FunctionTransformer(
                    _bool_to_float,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    if boolean_cols:
        transformers.append(("boolean", bool_pipeline, boolean_cols))

    return ColumnTransformer(transformers=transformers)


def build_model_pipeline(classifier, feature_sample: pd.DataFrame) -> Pipeline:
    """Full inference pipeline: feature engineering + preprocessing + classifier."""
    engineered_sample = FeatureEngineer().fit_transform(feature_sample)
    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", build_preprocessor(engineered_sample)),
            ("classifier", classifier),
        ]
    )
