# W4D4 — FastAPI Model Serving Endpoint

Train a scikit-learn Logistic Regression pipeline on the built-in breast cancer dataset, save with joblib, track evaluation in MLflow, and expose `/health` and `/predict` endpoints. Educational example only; not for clinical use.

## Windows CMD
```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
python -m pytest -v
uvicorn src.api:app --reload
```
Open http://127.0.0.1:8000/docs to try the endpoint. `/predict` requires a JSON object with `features`, a list of exactly 30 numbers in the original dataset feature order.

Run `python -m src.train` to generate `models/classifier.joblib` and `outputs/metrics.json`. Capture screenshots of the terminal tests and Swagger UI as evidence.

## Design decisions
- Use a Pipeline so preprocessing is identical at training and serving.
- Validate requests with Pydantic.
- Keep the model local; never load untrusted pickle/joblib files.
- Record experiments with MLflow.

## Scope
Lesson's surrounding text discusses class imbalance, but its named hands-on objective is FastAPI serving. This project focuses on a functioning serving API.
