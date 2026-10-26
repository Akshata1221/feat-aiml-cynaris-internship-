"""Sentiment capstone: compare Logistic Regression and Random Forest on text."""
from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, accuracy_score, precision_score, recall_score, roc_auc_score)

BASE = Path(__file__).resolve().parents[1]

def load_data():
    records = json.loads((BASE / "data" / "reviews.json").read_text(encoding="utf-8"))
    return [r["text"] for r in records], [r["label"] for r in records]

def train():
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    (BASE / "outputs").mkdir(exist_ok=True)
    (BASE / "models").mkdir(exist_ok=True)
    mlflow.set_tracking_uri((BASE / "mlruns").as_uri())
    mlflow.set_experiment("W4D5_Sentiment_Classifier")
    scores = {}
    for name, classifier in [("logistic", LogisticRegression(max_iter=1000, random_state=42)), ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42))]:
        pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("model", classifier)])
        pipeline.fit(Xtr, ytr)
        preds = pipeline.predict(Xte)
        probabilities = pipeline.predict_proba(Xte)[:, 1]
        metrics = {"accuracy": float(accuracy_score(yte, preds)), "precision": float(precision_score(yte, preds, zero_division=0)), "recall": float(recall_score(yte, preds, zero_division=0)), "roc_auc": float(roc_auc_score(yte, probabilities))}
        scores[name] = metrics
        print(f"\n{name} classification report:\n{classification_report(yte, preds, zero_division=0)}")
        ConfusionMatrixDisplay.from_predictions(yte, preds)
        plt.title(f"{name} confusion matrix")
        plt.tight_layout(); plt.savefig(BASE / "outputs" / f"{name}_confusion.png"); plt.close()
        RocCurveDisplay.from_predictions(yte, probabilities)
        plt.title(f"{name} ROC curve")
        plt.tight_layout(); plt.savefig(BASE / "outputs" / f"{name}_roc.png"); plt.close()
        joblib.dump(pipeline, BASE / "models" / f"{name}.joblib")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("classifier", name)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(BASE / "models" / f"{name}.joblib"))
    (BASE / "outputs" / "comparison.json").write_text(json.dumps(scores, indent=2))
    # Use ROC AUC to choose the model; tie-breaker prefers the simpler logistic model.
    best = max(["logistic", "random_forest"], key=lambda n: scores[n]["roc_auc"])
    joblib.dump(joblib.load(BASE / "models" / f"{best}.joblib"), BASE / "models" / "best_sentiment.joblib")
    print("Best model:", best)
    return scores

if __name__ == "__main__":
    train()
