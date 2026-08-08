from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"

PRIMARY_CANDIDATES = (
    "child_education_risk_intelligence.csv",
    "student_education_risk.csv",
    "education_risk_dataset.csv",
    "students.csv",
    "dataset.csv",
)
