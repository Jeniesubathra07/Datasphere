from fastapi import FastAPI

app = FastAPI(title="Datasphere", version="0.1.0")

DATASETS = [
    {"id": 1, "name": "sales", "records": 12840},
    {"id": 2, "name": "inventory", "records": 5021},
]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets() -> dict[str, list[dict[str, int | str]]]:
    return {"datasets": DATASETS}
