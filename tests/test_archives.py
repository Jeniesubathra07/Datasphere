import zipfile
from pathlib import Path

from datasphere.data.archives import extract_dataset_archives
from datasphere.data.loader import discover_datasets, load_primary_dataset


def test_extract_zip_and_load_csv(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "datasets" / "raw"
    raw_dir.mkdir(parents=True)
    archive = raw_dir / "student_education_risk.zip"
    csv_name = "student_education_risk.csv"

    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(csv_name, "student_id,attendance,test_score\n1,60,45\n2,95,88\n")

    import datasphere.data.paths as paths
    import datasphere.data.archives as archives
    import datasphere.data.loader as loader

    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(paths, "RAW_DIR", raw_dir)
    monkeypatch.setattr(archives, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(archives, "RAW_DIR", raw_dir)
    monkeypatch.setattr(loader, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(loader, "RAW_DIR", raw_dir)

    extracted = extract_dataset_archives()
    assert extracted == ["datasets/raw/student_education_risk.csv"]

    datasets = discover_datasets()
    assert datasets[0]["records"] == 2

    frame = load_primary_dataset()
    assert len(frame) == 2
