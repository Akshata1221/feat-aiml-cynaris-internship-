"""Optional MLflow tracking and Ragas evaluation helpers."""
def log_run(topic: str, mode: str, output_path: str) -> None:
    try:
        import mlflow
        from .config import settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("automated-research-report-agent")
        with mlflow.start_run():
            mlflow.log_param("topic", topic)
            mlflow.log_param("mode", mode)
            mlflow.log_artifact(output_path)
    except ImportError:
        print("MLflow not installed; skipped experiment tracking.")

def evaluate_with_ragas(dataset):
    """Evaluate a Ragas-compatible dataset; caller supplies references and model config."""
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError("Install ragas to run evaluation.") from exc
    return evaluate(dataset)
