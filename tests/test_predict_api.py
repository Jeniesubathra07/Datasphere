"""API prediction endpoint tests."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from datasphere.app import create_app
from datasphere.core.model_loader import load_production_model
from datasphere.data.paths import MODELS_DIR
from datasphere.ml.inference import build_synthetic_validation_student
from datasphere.data.loader import load_primary_dataset
from datasphere.ml.cleaning import clean_dataset


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_payload() -> dict:
    raw = clean_dataset(load_primary_dataset())
    student = build_synthetic_validation_student(raw)
    student["record_generated_date"] = str(student["record_generated_date"])[:10]
    student["enrollment_date"] = str(student["enrollment_date"])[:10]
    if isinstance(student.get("household_has_internet_access"), str):
        student["household_has_internet_access"] = student["household_has_internet_access"].lower() == "true"
    return student


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_model_info(client: TestClient) -> None:
    response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] == "logistic_regression"
    assert "test_metrics" in payload


def test_model_schema(client: TestClient) -> None:
    response = client.get("/api/model/schema")
    assert response.status_code == 200
    payload = response.json()
    assert "fields" in payload
    assert "dropout_probability_score" not in payload["fields"]
    assert "student_id" not in payload["fields"]


def test_predict_valid(client: TestClient, valid_payload: dict) -> None:
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["risk_level"] in {"Low", "Medium", "High", "Critical"}
    assert sum(payload["prediction"]["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert payload["model"]["version"]


def test_predict_explain(client: TestClient, valid_payload: dict) -> None:
    response = client.post("/predict/explain", json=valid_payload)
    assert response.status_code == 200
    payload = response.json()
    assert payload["explanation"] is not None
    assert payload["explanation"]["summary"]


def test_predict_missing_field(client: TestClient, valid_payload: dict) -> None:
    bad = dict(valid_payload)
    bad.pop("age")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_category(client: TestClient, valid_payload: dict) -> None:
    bad = dict(valid_payload)
    bad["gender"] = "InvalidGender"
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_numeric(client: TestClient, valid_payload: dict) -> None:
    bad = dict(valid_payload)
    bad["attendance_rate_pct"] = 150
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_extra_field_rejected(client: TestClient, valid_payload: dict) -> None:
    bad = dict(valid_payload)
    bad["dropout_probability_score"] = 0.5
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_model_unavailable(monkeypatch) -> None:
    from datasphere.core import model_loader

    def _fail(_path):
        raise model_loader.ModelLoadError("missing")

    monkeypatch.setattr(model_loader, "load_production_model", _fail)
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"age": 10})
        assert response.status_code in {422, 503}


def test_artifact_loads() -> None:
    loaded = load_production_model(MODELS_DIR)
    assert loaded.pipeline is not None
    assert loaded.metadata["selected_model"] == "logistic_regression"
