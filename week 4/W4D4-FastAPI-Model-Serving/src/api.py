"""FastAPI serving endpoint. Train first: python -m src.train."""
from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "classifier.joblib"
app = FastAPI(title="W4D4 Breast Cancer Classifier", version="1.0")

class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model missing. Run python -m src.train first.")
    return joblib.load(MODEL_PATH)  # Only load trusted locally generated artifacts.

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: PredictRequest):
    model = get_model()
    prediction = int(model.predict([payload.features])[0])
    confidence = float(model.predict_proba([payload.features])[0, prediction])
    return {"prediction": prediction, "label": "benign" if prediction == 1 else "malignant", "confidence": confidence}
