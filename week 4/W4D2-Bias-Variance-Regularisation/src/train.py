"""Compare regularisation, bias/variance, and search methods."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def make_pipeline(penalty="l2", C=1.0):
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=2000, random_state=42))])

def run():
    OUT.mkdir(exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(make_pipeline(), {"clf__penalty": ["l1", "l2"], "clf__C": [.01, .1, 1, 10, 100]}, cv=cv, scoring="f1", n_jobs=-1)
    random = RandomizedSearchCV(make_pipeline(), {"clf__penalty": ["l1", "l2"], "clf__C": np.logspace(-3, 3, 100)}, n_iter=12, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("W4D2_Bias_Variance_Regularisation")
    rows = []
    for name, search in [("grid", grid), ("random", random)]:
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            model = search.best_estimator_
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
            metrics = {"test_accuracy": accuracy_score(y_test, pred), "test_f1": f1_score(y_test, pred), "test_auc": roc_auc_score(y_test, prob), "cv_f1": search.best_score_, "train_accuracy": accuracy_score(y_train, model.predict(X_train))}
            mlflow.log_params({"method": name, **search.best_params_})
            mlflow.log_metrics(metrics)
            rows.append({"method": name, **search.best_params_, **metrics})
    pd.DataFrame(rows).to_csv(OUT / "search_comparison.csv", index=False)
    best = max([(grid.best_score_, grid.best_estimator_, "grid"), (random.best_score_, random.best_estimator_, "random")], key=lambda item: item[0])
    sizes, train, valid = learning_curve(best[1], X_train, y_train, cv=cv, scoring="accuracy", train_sizes=np.linspace(.15, 1, 6), n_jobs=-1)
    plt.plot(sizes, train.mean(axis=1), marker="o", label="Train")
    plt.plot(sizes, valid.mean(axis=1), marker="o", label="Cross-validation")
    plt.xlabel("Training samples"); plt.ylabel("Accuracy"); plt.title("Learning curve: bias vs variance"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "learning_curve.png", dpi=150); plt.close()
    with open(OUT / "best_model.json", "w") as f:
        json.dump({"method": best[2], "cv_f1": float(best[0])}, f, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    run()
