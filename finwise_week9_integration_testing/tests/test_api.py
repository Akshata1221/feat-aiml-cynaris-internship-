from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE = {
    "age": 47, "income_annual_inr": 840000, "employment_type": "self_employed",
    "employment_years": 20, "existing_loans": 1, "existing_emi_inr": 6772,
    "credit_score": 681, "loan_amount_inr": 500000, "loan_purpose": "medical",
    "ltv_ratio": 0.61, "dti_ratio": 0.15
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict_returns_probability_and_class():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in ("low", "medium", "high")

def test_predict_rejects_invalid_credit_score():
    bad = {**SAMPLE, "credit_score": 1200}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422

def test_predict_rejects_unknown_fields():
    bad = {**SAMPLE, "application_id": "APP-TEST"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
