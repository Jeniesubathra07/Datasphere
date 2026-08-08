from pathlib import Path

import pandas as pd

from datasphere.data.loader import discover_datasets
from datasphere.ml.labels import construct_risk_label
from datasphere.ml.train import train_model


def test_discover_datasets_with_sample(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "datasets" / "raw"
    raw_dir.mkdir(parents=True)
    sample = raw_dir / "student_education_risk.csv"
    sample.write_text("student_id,attendance,test_score\n1,60,45\n2,95,88\n", encoding="utf-8")

    import datasphere.data.paths as paths
    import datasphere.data.loader as loader

    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(paths, "RAW_DIR", raw_dir)
    monkeypatch.setattr(loader, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(loader, "RAW_DIR", raw_dir)

    datasets = discover_datasets()
    assert len(datasets) == 1
    assert datasets[0]["name"] == "student_education_risk"


def test_construct_risk_label() -> None:
    df = pd.DataFrame(
        {
            "attendance": [50, 90],
            "test_score": [30, 85],
            "child_labour": ["yes", "no"],
            "household_income": [10000, 50000],
        }
    )
    labels = construct_risk_label(df)
    assert labels.iloc[0] in {"High", "Critical", "Medium"}
    assert labels.iloc[1] in {"Low", "Medium"}


def test_train_model_on_sample() -> None:
    df = pd.DataFrame(
        {
            "attendance": [50, 55, 90, 92, 40, 45, 88, 91] * 5,
            "test_score": [30, 35, 85, 88, 25, 28, 82, 86] * 5,
            "child_labour": ["yes", "yes", "no", "no", "yes", "yes", "no", "no"] * 5,
            "household_income": [10000, 12000, 50000, 52000, 9000, 11000, 48000, 51000] * 5,
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"] * 5,
        }
    )
    result = train_model(df)
    assert result["rows"] == 40
    assert Path(result["model_path"]).exists()
