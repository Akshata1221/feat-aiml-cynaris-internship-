from src.train import make_pipeline
from sklearn.datasets import load_breast_cancer

def test_pipeline_predictions():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline("l1", 0.1).fit(X[:400], y[:400])
    assert len(model.predict(X[400:410])) == 10
    assert model.predict_proba(X[400:410]).shape == (10, 2)

def test_l2_supported():
    assert make_pipeline("l2").named_steps["clf"].penalty == "l2"
