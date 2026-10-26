from fastapi.testclient import TestClient
from src.train import load_data, train
from src.api import app, MODEL, get_model

def test_dataset():
    X, y = load_data()
    assert len(X) >= 40 and len(X) == len(y)
    assert set(y) == {0, 1}

def test_api_validation():
    assert TestClient(app).post("/predict", json={"text": ""}).status_code == 422

def test_training_and_prediction():
    if not MODEL.exists():
        scores = train()
        assert set(scores) == {"logistic", "random_forest"}
    get_model.cache_clear()
    result = TestClient(app).post("/predict", json={"text": "Wonderful quality and great service"})
    assert result.status_code == 200
    assert result.json()["sentiment"] in {"positive", "negative"}
