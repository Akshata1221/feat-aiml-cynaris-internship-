import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate_models import metric_values, make_models
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def test_perfect_predictions():
    result = metric_values([0, 1, 0, 1], [0, 1, 0, 1], [.1, .9, .2, .8])
    assert all(abs(value - 1.0) < 1e-10 for value in result.values())


def test_models_fit_and_predict_probability():
    X, y = load_breast_cancer(return_X_y=True)
    for model in make_models().values():
        model.fit(X[:100], y[:100])
        assert model.predict_proba(X[100:110]).shape == (10, 2)


def test_stratified_cv_runs():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    result = cross_val_score(make_models()['LogisticRegression'], X_train, y_train, cv=cv)
    assert len(result) == 5
    assert (result >= 0).all() and (result <= 1).all()
