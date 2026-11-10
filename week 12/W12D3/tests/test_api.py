from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_prediction():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1, 2]
    assert body["class_name"] in ["setosa", "versicolor", "virginica"]

def test_reject_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422

def test_reject_non_finite_values():
    response = client.post("/predict", json={"features": [float("nan"), 3.5, 1.4, 0.2]})
    assert response.status_code == 422
