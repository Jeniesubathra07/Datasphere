from __future__ import annotations

import zipfile
from pathlib import Path

from datasphere.data.paths import DATASETS_DIR, RAW_DIR


def _zip_files(*directories: Path) -> list[Path]:
    archives: list[Path] = []
    for directory in directories:
        if directory.exists():
            archives.extend(sorted(directory.glob("*.zip")))
    return archives


def extract_dataset_archives(force: bool = False) -> list[str]:
    """Extract dataset zip archives into datasets/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []

    for archive in _zip_files(DATASETS_DIR, RAW_DIR):
        with zipfile.ZipFile(archive, "r") as zipped:
            for member in zipped.namelist():
                if member.endswith("/"):
                    continue
                destination = RAW_DIR / Path(member).name
                if destination.exists() and not force:
                    continue
                with zipped.open(member) as source, destination.open("wb") as target:
                    target.write(source.read())
                extracted.append(str(destination.relative_to(DATASETS_DIR.parent)))

    return extracted
