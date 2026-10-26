# W4D3-Model-Serialisation: Model Serialisation — joblib & pickle

## Goal
Save and reload a complete trained pipeline using joblib and pickle, verify predictions, and evaluate classification metrics.

Uses scikit-learn for model training and **MLflow** for experiment tracking (approved stack). CrewAI, LangGraph and Ragas are not needed for this supervised learning task.

## Windows CMD setup

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python src\serialize.py
mlflow ui --backend-store-uri ./mlruns
```

Open the MLflow UI at http://127.0.0.1:5000. Output files are generated in `outputs/` after running; they are not precomputed evidence. Record screenshots after execution.

## Deliverables
- Python code, tests, MLflow logs, generated CSV/JSON/plots
- `docs/self_review.md`, `docs/cia_interactions.md`, `docs/viva_answers.md`, `docs/git_steps.md`
- Branch `feat/aiml-W4-Akshata`, 2+ commits and GitHub PR

## Notes
Use Python 3.11–3.13 for broad dependency compatibility. Training and test data are separated; preprocessing is inside pipelines. Do not load untrusted pickle or joblib artifacts.
