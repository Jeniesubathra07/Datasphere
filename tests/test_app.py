from fastapi.testclient import TestClient

from datasphere.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_datasets() -> None:
    response = client.get("/api/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["datasets"]) == 2
    assert payload["datasets"][0]["name"] == "sales"
