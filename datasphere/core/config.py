"""Application configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from datasphere.data.paths import MODELS_DIR, REPO_ROOT

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

MODELS_PATH: Path = Path(os.getenv("MODELS_DIR", str(MODELS_DIR)))
API_PREFIX: str = os.getenv("API_PREFIX", "")
APP_VERSION: str = "3.0.0"
REPO_ROOT_PATH: Path = REPO_ROOT
