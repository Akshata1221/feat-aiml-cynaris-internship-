import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PARAM_SETS = [
    {"n_estimators": 25, "max_depth": 2},
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 75, "max_depth": 4},
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 150, "max_depth": 5},
]

def train_and_register(experiment="W11D2-Registry", model_name="IrisClassifier"):
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )
    mlflow.set_experiment(experiment)
    client = MlflowClient()
    runs = []
    for i, params in enumerate(PARAM_SETS, 1):
        with mlflow.start_run(run_name=f"candidate-{i}") as run:
            model = RandomForestClassifier(**params, random_state=42).fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", score)
            info = mlflow.sklearn.log_model(model, name="model")
            runs.append((score, run.info.run_id, info.model_uri))
    score, run_id, model_uri = max(runs, key=lambda item: item[0])
    registered = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, "champion", str(registered.version))
    print(f"Registered {model_name} version={registered.version}, accuracy={score:.4f}")
    loaded = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    print("Loaded model prediction:", loaded.predict(X_test.head(1)).tolist())
    return {"run_id": run_id, "accuracy": score, "version": registered.version}

if __name__ == "__main__":
    train_and_register()
