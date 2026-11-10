# Ragas Evaluation + MLOps Quality Checks

Provide a small research-report pipeline with automated unit tests, reproducible offline mode, MLflow run logging and an optional Ragas evaluation adapter.

## Setup

```bash
python -m venv .venv
# activate .venv
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
python -m research_agent.cli --topic "AI governance" --mode offline
```
For Ragas evaluation, supply a Ragas-compatible dataset to `research_agent.tracking.evaluate_with_ragas(dataset)`. Evaluation model/embedding configuration is intentionally caller-supplied.

## Repository contents

This standalone folder contains the shared report models, search adapter, CLI and documentation needed by the focused implementation. Optional integrations are imported only when used. See `docs/ARCHITECTURE.md` and `docs/VIVA.md` for design and review notes.

## Configuration and safety

Copy `.env.example` to `.env` only when needed. Never commit credentials. Offline mode is deterministic and does not claim live research.
