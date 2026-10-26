# W5D2 — Prompt Engineering with Ollama

Run five prompts with three system-prompt styles and compare `llama3.2:3b` and `qwen2.5:3b` on the same three questions. Results are recorded, not fabricated.

## Windows CMD setup
1. Install Ollama from https://ollama.com/download/windows
2. `ollama pull llama3.2:3b`
3. `ollama pull qwen2.5:3b`
4. Ensure Ollama is running (if necessary: `ollama serve` in a separate terminal).
5. `py -m venv .venv`
6. `.venv\Scripts\activate`
7. `pip install -r requirements.txt`
8. `python -m pytest -v`
9. `python -m src.prompt_experiment --mlflow`

Inspect `outputs/responses.json` and `outputs/comparison.csv`. Complete `docs/comparison_review.md` with real observations. The experiment uses temperature 0.2 and a fixed seed to improve repeatability, but generation may still vary. No CrewAI, LangGraph, or Ragas components are necessary for this local prompting exercise; MLflow is included.

## Git
Create a new GitHub repository for W5D2, then run `git init`, `git checkout -b feat/aiml-W5-Akshata`, `git add .`, `git commit -m "feat: ollama setup — local llm inference"`, and after running and reviewing evidence, `git add .` and `git commit -m "docs: document prompt comparisons and tests"`. Add your GitHub remote and `git push -u origin feat/aiml-W5-Akshata`; open a PR into main. Complete two real CIA mentor interactions **before** your first commit. Output files are gitignored by default; if submission requires evidence, selectively add a redacted CSV using `git add -f outputs/comparison.csv`.
