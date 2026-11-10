# MLflow Experiment Tracking for Research Reports

Track research runs with MLflow parameters and report artifacts. Each run records its topic and execution mode under a dedicated experiment.

## Run
```bash
pip install -r requirements.txt
python -m research_agent.cli --topic "AI safety" --mode offline
mlflow ui --backend-store-uri ./mlruns
```
Open the local MLflow UI URL printed by the command. Run records and artifacts are stored locally in `mlruns/`.
