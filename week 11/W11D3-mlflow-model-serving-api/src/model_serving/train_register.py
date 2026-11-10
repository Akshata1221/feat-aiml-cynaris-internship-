import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_register(model_name="IrisServingModel"):
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=7, stratify=iris.target
    )
    mlflow.set_experiment("W11D3-Serving")
    with mlflow.start_run(run_name="serving-candidate") as run:
        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=7)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_params({"n_estimators": 100, "max_depth": 4})
        mlflow.log_metric("accuracy", score)
        model_info = mlflow.sklearn.log_model(model, name="model")
    registered = mlflow.register_model(model_info.model_uri, model_name)
    mlflow.set_registered_model_alias(model_name, "champion", str(registered.version))
    return {"model_name": model_name, "version": registered.version, "accuracy": score}

if __name__ == "__main__":
    print(train_register())
