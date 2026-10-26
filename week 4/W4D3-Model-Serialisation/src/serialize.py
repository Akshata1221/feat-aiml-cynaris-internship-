"""Train, evaluate, save and safely verify local model artifacts."""
from pathlib import Path
import json
import pickle
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import mlflow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def make_model():
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))])

def save_models(model, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    with (folder / "model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def run():
    OUT.mkdir(exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = make_model().fit(X_train, y_train)
    save_models(model, OUT)
    # Never load pickle/joblib files from untrusted sources.
    loaded_joblib = joblib.load(OUT / "model.joblib")
    with (OUT / "model.pkl").open("rb") as f:
        loaded_pickle = pickle.load(f)
    original = model.predict(X_test)
    assert (original == loaded_joblib.predict(X_test)).all()
    assert (original == loaded_pickle.predict(X_test)).all()
    probabilities = loaded_joblib.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, original), "precision": precision_score(y_test, original), "recall": recall_score(y_test, original), "f1": f1_score(y_test, original), "roc_auc": roc_auc_score(y_test, probabilities)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual": y_test, "prediction": original, "probability": probabilities}).to_csv(OUT / "predictions.csv", index=False)
    for name, plotter in [("confusion_matrix", lambda: ConfusionMatrixDisplay.from_predictions(y_test, original)), ("roc_curve", lambda: RocCurveDisplay.from_predictions(y_test, probabilities)), ("precision_recall_curve", lambda: PrecisionRecallDisplay.from_predictions(y_test, probabilities))]:
        plotter(); plt.tight_layout(); plt.savefig(OUT / f"{name}.png", dpi=150); plt.close()
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("W4D3_Model_Serialisation")
    with mlflow.start_run(run_name="joblib_pickle_comparison"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("formats", "joblib,pickle")
        mlflow.log_artifact(str(OUT / "metrics.json"))
    print("Saved both formats; prediction equivalence verified.", metrics)

if __name__ == "__main__":
    run()
