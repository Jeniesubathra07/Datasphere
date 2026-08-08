from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasphere.data.archives import extract_dataset_archives
from datasphere.data.paths import DATASETS_DIR, PRIMARY_CANDIDATES, RAW_DIR


def _csv_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


def _zip_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.zip"))


def ensure_datasets_extracted() -> list[str]:
    return extract_dataset_archives()


def discover_datasets() -> list[dict[str, str | int]]:
    ensure_datasets_extracted()
    files = _csv_files(RAW_DIR) or _csv_files(DATASETS_DIR)
    datasets: list[dict[str, str | int]] = []
    for index, path in enumerate(files, start=1):
        try:
            row_count = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
        except OSError:
            row_count = 0
        datasets.append(
            {
                "id": index,
                "name": path.stem,
                "path": str(path.relative_to(DATASETS_DIR.parent)),
                "records": max(row_count, 0),
            }
        )

    if not datasets:
        for index, path in enumerate(_zip_files(RAW_DIR) or _zip_files(DATASETS_DIR), start=1):
            datasets.append(
                {
                    "id": index,
                    "name": path.stem,
                    "path": str(path.relative_to(DATASETS_DIR.parent)),
                    "records": 0,
                    "format": "zip",
                }
            )
    return datasets


def resolve_primary_dataset() -> Path | None:
    ensure_datasets_extracted()

    for name in PRIMARY_CANDIDATES:
        candidate = RAW_DIR / name
        if candidate.exists():
            return candidate

    files = _csv_files(RAW_DIR) or _csv_files(DATASETS_DIR)
    return files[0] if files else None


def load_primary_dataset() -> pd.DataFrame:
    path = resolve_primary_dataset()
    if path is None:
        raise FileNotFoundError(
            "No dataset found. Add a CSV or zip under datasets/raw/ "
            "(for example datasets/raw/student_education_risk.zip)."
        )
    return pd.read_csv(path)
