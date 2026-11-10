# W11D1 — MLflow Experiment Tracking & Logging

## Objective
Train an sklearn Iris classifier across five hyperparameter configurations and log parameters, metrics, and model artifacts to MLflow.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

## Run
```bash
mlflow ui --host 127.0.0.1 --port 5000
python -m mlflow_tracking.train
```
Open http://127.0.0.1:5000 to compare the five runs.

## Structure
- `src/mlflow_tracking/` — implementation
- `tests/` — automated tests
- `docs/SELF_REVIEW.md` — submission checklist
- `docs/VIVA.md` — viva preparation

## Git submission
```bash
git init
git switch -c feat/aiml-W11-akshata
git add .
git commit -m "feat: implement MLflow or RAG workflow"
git add README.md docs tests
git commit -m "test: add workflow coverage and submission docs"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin feat/aiml-W11-akshata
```
Create a pull request on GitHub.

## Submission limitations
This archive does not create remote GitHub commits, push a branch, open a PR, record CIA interactions, or mark the LMS lesson complete. Complete those platform actions yourself. MLflow registry/serving and Ragas scoring require the optional services and dependencies described above.
