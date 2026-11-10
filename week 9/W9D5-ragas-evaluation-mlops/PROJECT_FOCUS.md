# Ragas Evaluation + MLOps Quality Checks

Provide a small research-report pipeline with automated unit tests, reproducible offline mode, MLflow run logging and an optional Ragas evaluation adapter.

## Run
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
python -m research_agent.cli --topic "AI governance" --mode offline
```
For Ragas evaluation, supply a Ragas-compatible dataset to `research_agent.tracking.evaluate_with_ragas(dataset)`. Evaluation model/embedding configuration is intentionally caller-supplied.
