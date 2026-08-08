from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def split_features_target(df: pd.DataFrame, target_col: str = "risk_level") -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_col], errors="ignore")
    target = df[target_col]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
