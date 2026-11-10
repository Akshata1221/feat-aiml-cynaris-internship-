import argparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

CONFIGS = [
    {"n_estimators": 30, "max_depth": 2, "min_samples_split": 2},
    {"n_estimators": 60, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 4, "min_samples_split": 2},
    {"n_estimators": 120, "max_depth": None, "min_samples_split": 3},
    {"n_estimators": 160, "max_depth": 5, "min_samples_split": 4},
]

def run_experiments(experiment_name="W11D1-Iris-Experiments"):
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    mlflow.set_experiment(experiment_name)
    results = []
    for index, params in enumerate(CONFIGS, start=1):
        with mlflow.start_run(run_name=f"random-forest-{index}") as run:
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="weighted")
            mlflow.log_params({k: str(v) if v is None else v for k, v in params.items()})
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("weighted_f1", f1)
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, name="iris_classifier", signature=signature,
                input_example=X_train.head(3)
            )
            results.append({"run_id": run.info.run_id, "accuracy": accuracy, "weighted_f1": f1, **params})
            print(f"run={index} accuracy={accuracy:.4f} weighted_f1={f1:.4f}")
    return sorted(results, key=lambda row: row["accuracy"], reverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="W11D1-Iris-Experiments")
    args = parser.parse_args()
    print("Ranked runs:", run_experiments(args.experiment))
