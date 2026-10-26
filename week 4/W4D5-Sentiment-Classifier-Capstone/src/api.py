from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = Path(__file__).resolve().parents[1] / "models" / "best_sentiment.joblib"
app = FastAPI(title="W4D5 Sentiment Classifier")

class Review(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

@lru_cache(maxsize=1)
def get_model():
    if not MODEL.exists():
        raise HTTPException(503, "Train the model first: python -m src.train")
    return joblib.load(MODEL)  # Load only locally generated trusted model files.

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL.exists()}

@app.post("/predict")
def predict(review: Review):
    model = get_model()
    label = int(model.predict([review.text])[0])
    return {"sentiment": "positive" if label == 1 else "negative", "positive_probability": float(model.predict_proba([review.text])[0, 1])}
