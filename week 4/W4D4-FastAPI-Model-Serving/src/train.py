"""Train a reproducible binary classifier and persist the complete preprocessing pipeline."""
from pathlib import Path
import json
import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parents[1]
MODEL = BASE / "models" / "classifier.joblib"

def train():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "roc_auc": roc_auc_score(y_test, prob)}
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL)
    (BASE / "outputs").mkdir(exist_ok=True)
    (BASE / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    mlflow.set_tracking_uri((BASE / "mlruns").as_uri())
    mlflow.set_experiment("W4D4_FastAPI_Model_Serving")
    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL))
    print(metrics)
    return pipeline

if __name__ == "__main__":
    train()
