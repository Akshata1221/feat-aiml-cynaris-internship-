from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from src.api import app, get_model
from src.train import train, MODEL

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_validation():
    assert client.post("/predict", json={"features": [1.0]}).status_code == 422

def test_prediction():
    if not MODEL.exists():
        train()
    get_model.cache_clear()
    sample = load_breast_cancer().data[0].tolist()
    result = client.post("/predict", json={"features": sample})
    assert result.status_code == 200
    assert result.json()["prediction"] in [0, 1]
    assert 0 <= result.json()["confidence"] <= 1
