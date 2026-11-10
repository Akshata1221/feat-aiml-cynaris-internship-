"""Train and persist a deterministic Iris classifier."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "model.joblib"

def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved {OUT}; holdout_accuracy={score:.4f}")

if __name__ == "__main__":
    main()
