from fastapi.testclient import TestClient

from datasphere.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["platform"] == "EduRisk Intelligence"


def test_list_datasets() -> None:
    response = client.get("/api/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert "datasets" in payload
    assert isinstance(payload["datasets"], list)
