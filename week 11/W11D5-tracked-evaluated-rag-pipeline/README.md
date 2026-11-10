# W11D5 — Tracked & Evaluated RAG Pipeline

## Objective
End-to-end RAG workflow with a local document corpus, retrieval, MLflow run tracking, ten evaluation questions, and optional Ragas metrics.

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
python -m tracked_rag.pipeline
```
This prepares ten reference Q&A samples and logs an MLflow run when MLflow is installed. For Ragas scoring, install `pip install -e '.[evaluation]'` and configure an evaluation LLM/embedding provider.

## Structure
- `src/tracked_rag/` — implementation
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
