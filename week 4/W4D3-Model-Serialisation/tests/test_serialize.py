from src.serialize import make_model, save_models
from sklearn.datasets import load_breast_cancer
import joblib
import pickle

def test_save_and_reload(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    model = make_model().fit(X[:400], y[:400])
    save_models(model, tmp_path)
    a = joblib.load(tmp_path / "model.joblib")
    with (tmp_path / "model.pkl").open("rb") as f:
        b = pickle.load(f)
    assert (model.predict(X[400:]) == a.predict(X[400:])).all()
    assert (model.predict(X[400:]) == b.predict(X[400:])).all()

def test_pipeline_contains_scaler():
    assert "scaler" in make_model().named_steps
