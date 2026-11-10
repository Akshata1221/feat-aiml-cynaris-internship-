"""FastAPI inference service with a small, reproducible scikit-learn model."""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        examples=[[5.1, 3.5, 1.4, 0.2]]
    )

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model artifact missing. Run: python scripts/train.py")
    return joblib.load(MODEL_PATH)

app = FastAPI(title="Production ML Prediction API", version="1.0.0")

@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/ready")
def ready():
    return health()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    values = np.asarray(request.features, dtype=float).reshape(1, -1)
    if not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")
    try:
        model = load_model()
        prediction = int(model.predict(values)[0])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return PredictionResponse(
        prediction=prediction,
        class_name=names.get(prediction, "unknown"),
        model_version="iris-rf-v1"
    )
