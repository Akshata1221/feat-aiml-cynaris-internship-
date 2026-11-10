# Automated Research Report Agent

A submission-ready Week 9 project combining **CrewAI, LangChain, LangGraph, MLflow, Ragas and MLOps practices**.

It builds a research report through a three-agent pipeline (Researcher → Writer → Reviewer), supports live DuckDuckGo search, records runs with MLflow, exposes an optional Ollama/LangChain conversational chain, and includes an optional LangGraph orchestration path. A deterministic offline mode makes the project testable without API keys.

## Features
- Three explicitly configured CrewAI agents with role, goal and backstory.
- Three sequential CrewAI tasks and a web-search tool.
- Offline baseline vs. sourced research mode for documenting quality improvement.
- LangGraph pipeline orchestration (optional dependency).
- LangChain + Ollama chain and conversation history helper (optional dependency).
- MLflow run tracking and optional Ragas evaluation hooks.
- CLI, tests, sample output, self-review checklist and viva answers.

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m research_agent.cli --topic "How generative AI is changing software testing" --mode offline
```

## Run with live search + CrewAI
1. Add your model provider key and `SERPER_API_KEY` (if using Serper) or configure the DuckDuckGo tool in `.env`.
2. Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`.
3. Run:
```bash
python -m research_agent.cli --topic "Your research topic" --mode crew
```
The default live-search implementation uses DuckDuckGo and does not require a Serper key. The CrewAI run requires a supported LLM provider credential.

## Run tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Project structure
```text
src/research_agent/
  config.py          Environment configuration
  models.py          Typed report and source models
  search.py          Live search + safe offline fallback
  pipeline.py        Baseline research/report pipeline
  crew.py            CrewAI agents, tasks and crew
  graph.py           Optional LangGraph orchestration
  hybrid.py          Optional LangChain + Ollama memory chain
  tracking.py        MLflow and Ragas integrations
  cli.py             Command-line entry point
tests/
  test_pipeline.py
  test_search.py
outputs/             Generated reports and run evidence
docs/                Architecture, improvement notes, viva and checklist
```

## Configuration
See `.env.example`. Never commit API keys. `.env` is ignored by Git.

## Evidence
Run the offline command and save the terminal output or generated Markdown report as evidence. For a live run, retain the source list and report metadata. Do not fabricate screenshots or claim a live search if the offline fallback was used.

## Git submission workflow
Replace `akshata` in the branch name if your course requires a different spelling:
```bash
git init
git checkout -b feat/aiml-W9-akshata
git add .
git commit -m "feat: add research report pipeline and tests"
git add .
git commit -m "docs: document agent workflow and evaluation"
git push -u origin feat/aiml-W9-akshata
```
Then open a Pull Request against the course repository's required base branch.

## CIA / mentor review evidence
This ChatGPT session can help review the code, but it cannot create entries in your course's CIA evaluation log. Record two actual interactions in CIA Full Stack Mentor Mode separately:
1. Architecture/code review of the three-agent pipeline.
2. Review of test results, error handling and Git diff before PR.

## Evaluation note
The offline mode validates project structure and produces a reproducible example. For a full rubric submission, run the live CrewAI mode with your own credentials, capture real output, complete the self-review checklist and submit a PR.
