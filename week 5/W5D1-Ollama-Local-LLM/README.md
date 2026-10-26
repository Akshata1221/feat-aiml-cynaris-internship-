# W5D1 — Running LLMs Locally with Ollama

This project calls the local Ollama HTTP API with a custom system prompt, tests five prompts using `llama3.2:3b`, and compares three identical questions against `qwen2.5:3b`. It exports response text and latency to CSV/JSON, and optionally logs latency to MLflow. Quality must be evaluated manually; speed alone does not measure response quality.

## Windows setup (Command Prompt)

1. Install Ollama from https://ollama.com/download/windows and open Ollama.
2. In CMD, check `ollama --version` and run:

```bat
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama run llama3.2:3b
```

Try a question and enter `/bye` to exit interactive mode. Both model downloads require disk space and internet; inference runs locally. If `ollama` is not recognized, reopen CMD after installing. Ensure Ollama is running; if needed run `ollama serve` in a separate CMD window (only if the server is not already active).

3. Open this folder in VS Code, then:

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -v
python -m src.ollama_experiment --mlflow
```

Omit `--mlflow` to skip experiment tracking. Open the dashboard with `mlflow ui` (usually http://127.0.0.1:5000). The script makes eight total inference calls; CPU-only execution may be slow. It requires both model names to be installed.

## Outputs and evidence

- `outputs/responses.json`: full model responses and timings
- `outputs/comparison.csv`: response comparison table
- Screenshots: Ollama model list, terminal inference, passing pytest tests, and MLflow experiment (if used)
- Fill `docs/comparison_review.md` with your own observations from real outputs. Do not invent results.

## Design choices

- Same three prompts and deterministic temperature=0 for a fairer comparison.
- HTTP timeout and errors make setup problems easier to diagnose.
- Local API calls do not send prompts to a hosted model provider, though downloaded models originate online.
- MLflow tracks reproducible latency measurements; CrewAI, LangGraph, and Ragas are not needed for a basic single-model inference lesson and are not claimed as implemented.

## Git and submission

See `docs/git_steps.md`. Record two actual CIA Full Stack Mentor interactions and request review **before** your first commit. Then create at least two commits, push the feature branch, open a PR, and include evidence and the self-review checklist.
