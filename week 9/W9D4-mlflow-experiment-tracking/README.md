# MLflow Experiment Tracking for Research Reports

Track research runs with MLflow parameters and report artifacts. Each run records its topic and execution mode under a dedicated experiment.

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
python -m research_agent.cli --topic "AI safety" --mode offline
mlflow ui --backend-store-uri ./mlruns
```
Open the local MLflow UI URL printed by the command. Run records and artifacts are stored locally in `mlruns/`.

## Repository contents

This standalone folder contains the shared report models, search adapter, CLI and documentation needed by the focused implementation. Optional integrations are imported only when used. See `docs/ARCHITECTURE.md` and `docs/VIVA.md` for design and review notes.

## Configuration and safety

Copy `.env.example` to `.env` only when needed. Never commit credentials. Offline mode is deterministic and does not claim live research.
