from pathlib import Path

import pandas as pd
import pytest

from datasphere.data.loader import discover_datasets
from datasphere.ml.labels import construct_risk_label
from datasphere.ml.pipeline import run_stage2_pipeline


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


@pytest.mark.slow
def test_train_model_on_sample(tmp_path: Path) -> None:
    from datasphere.data.loader import load_primary_dataset

    df = load_primary_dataset().sample(n=1500, random_state=42)
    result = run_stage2_pipeline(df, models_dir=tmp_path, run_tuning=False)
    assert result.x_shape[0] == 1500
    assert Path(result.artifact_paths["model"]).exists()
