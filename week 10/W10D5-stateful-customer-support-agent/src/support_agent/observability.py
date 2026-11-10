def log_run(inputs: dict, outputs: dict, experiment="w10-customer-support"):
    """Optional MLflow tracking. Install mlflow to enable."""
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_param("intent", inputs.get("intent", "unknown"))
        mlflow.log_param("status", outputs.get("status", "unknown"))
        mlflow.log_metric("response_chars", len(outputs.get("response", "")))
    return True
