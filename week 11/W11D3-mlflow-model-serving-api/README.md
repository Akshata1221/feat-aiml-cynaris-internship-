# W11D3 — MLflow Model Serving & REST API

## Objective
Train and register a sklearn classifier, then serve the registered model through MLflow's REST API and call its predictions endpoint.

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
python -m model_serving.train_register
mlflow models serve -m "models:/IrisServingModel@champion" -p 5001 --env-manager local
python -m model_serving.client
```
Run the serving command in a second terminal after registration. REST input uses the MLflow `dataframe_split` format.

## Structure
- `src/model_serving/` — implementation
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
