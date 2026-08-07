from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from datasphere.data.paths import MODELS_DIR
from datasphere.ml.features import build_preprocessor, split_features_target
from datasphere.ml.labels import RISK_LEVELS, construct_risk_label


def prepare_labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["risk_level"] = construct_risk_label(labeled)
    return labeled


def train_model(df: pd.DataFrame, model_path: Path | None = None) -> dict[str, object]:
    labeled = prepare_labeled_frame(df)
    features, target = split_features_target(labeled)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target.astype(str))

    split_kwargs: dict[str, object] = {
        "test_size": 0.2,
        "random_state": 42,
    }
    if len(set(encoded_target)) > 1:
        split_kwargs["stratify"] = encoded_target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        **split_kwargs,
    )

    preprocessor = build_preprocessor(features)
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    raw_predictions = pipeline.predict(x_test)
    if getattr(raw_predictions, "ndim", 1) > 1:
        raw_predictions = raw_predictions.argmax(axis=1)
    predictions = label_encoder.inverse_transform(raw_predictions.astype(int))
    y_test_labels = label_encoder.inverse_transform(y_test)

    report = classification_report(y_test_labels, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test_labels, predictions, labels=label_encoder.classes_).tolist()

    destination = model_path or (MODELS_DIR / "education_risk_model.joblib")
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, destination)

    return {
        "model_path": str(destination),
        "rows": len(labeled),
        "features": len(features.columns),
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": label_encoder.classes_.tolist(),
    }
