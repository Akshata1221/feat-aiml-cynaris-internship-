"""FastAPI serving layer for FinWise loan default risk model."""
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "best_model.joblib"))
MODEL = joblib.load(MODEL_PATH)

app = FastAPI(title="FinWise Credit Risk API", version="1.0.0",
              description="Predicts loan default risk from applicant and loan features.")

class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=18, le=100)
    income_annual_inr: float = Field(gt=0)
    employment_type: str
    employment_years: float = Field(ge=0, le=80)
    existing_loans: int = Field(ge=0)
    existing_emi_inr: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=900)
    loan_amount_inr: float = Field(gt=0)
    loan_purpose: str
    ltv_ratio: float = Field(ge=0)
    dti_ratio: float = Field(ge=0)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(payload: LoanApplication):
    try:
        row = pd.DataFrame([payload.model_dump()])
        probability = float(MODEL.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        return {
            "prediction": prediction,
            "default_probability": round(probability, 6),
            "risk_band": "high" if probability >= 0.7 else ("medium" if probability >= 0.3 else "low"),
            "model": "random_forest_rf_baseline",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}")
