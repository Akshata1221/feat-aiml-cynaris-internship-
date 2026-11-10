# W11D4 — RAG Evaluation with Ragas

## Objective
Build a small LangChain + Chroma RAG pipeline, prepare ten evaluation samples, and run Ragas metrics for faithfulness, answer relevancy, context precision, and context recall.

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
The RAG dependencies and Ragas evaluator are optional and may require a compatible local Ollama embedding model plus an LLM configured for Ragas. Start with `pip install -e '.[rag,evaluation]'`. The code deliberately keeps retrieval and evaluation separate so you can swap the model provider.

## Structure
- `src/rag_evaluation/` — implementation
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
