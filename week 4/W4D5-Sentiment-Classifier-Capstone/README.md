# W4D5 — Sentiment Classifier: Deploy & Document

This capstone trains Logistic Regression and Random Forest binary sentiment classifiers, prints classification reports, compares accuracy/precision/recall/ROC-AUC, saves confusion matrices and ROC plots, and serves the selected model using FastAPI.

**Dataset:** `data/reviews.json` is a small, handcrafted **synthetic educational dataset**. Metrics are demonstration-only and do not represent real-world generalization. Replace it with a larger labeled public dataset for meaningful performance claims.

## Windows CMD setup
```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
python -m pytest -v
uvicorn src.api:app --reload
```
Open http://127.0.0.1:8000/docs and submit `{"text":"I really enjoyed this product"}` to `/predict`.

Outputs are generated in `outputs/`; model artifacts are in `models/`. MLflow local experiment runs are stored in `mlruns/`.

## Why these models?
TF-IDF converts text to numeric features without fitting on the test set. Logistic Regression is a strong linear baseline for sparse text, while Random Forest provides a nonlinear comparison. A stratified split preserves label proportions.

## Required commit message
Use `feat: classification — logistic vs random forest comparison` as your first commit message after obtaining CIA review. A second commit should document outputs and tests.
