from fastapi import FastAPI, HTTPException

from datasphere.data.loader import discover_datasets, load_primary_dataset
from datasphere.ml.train import train_model

app = FastAPI(
    title="EduRisk Intelligence",
    version="0.2.0",
    description="Child Education Risk Intelligence Platform API",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "platform": "EduRisk Intelligence"}


@app.get("/api/datasets")
def list_datasets() -> dict[str, object]:
    datasets = discover_datasets()
    if not datasets:
        return {
            "datasets": [],
            "message": "No CSV files found. Add datasets under datasets/raw/ on GitHub.",
        }
    return {"datasets": datasets}


@app.post("/api/model/train")
def train_risk_model() -> dict[str, object]:
    try:
        frame = load_primary_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return train_model(frame)
