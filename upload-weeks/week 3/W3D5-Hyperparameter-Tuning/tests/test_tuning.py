import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tune_models import make_searches, run
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def test_four_searches():
    searches = make_searches()
    assert len(searches) == 4
    assert sum(isinstance(s, GridSearchCV) for s in searches.values()) == 2
    assert sum(isinstance(s, RandomizedSearchCV) for s in searches.values()) == 2


def test_random_search_is_bounded():
    searches = make_searches(n_iter=4)
    assert searches['SVM_RandomSearch'].n_iter == 4
    assert searches['KNN_RandomSearch'].n_iter == 4


def test_end_to_end(tmp_path):
    result = run(output_dir=tmp_path, cv_folds=2, n_iter=2, use_mlflow=False)
    assert len(result) == 4
    assert result['accuracy'].between(0, 1).all()
    assert (tmp_path / 'model_comparison.csv').exists()
    assert (tmp_path / 'best_model.json').exists()
    assert len(list(tmp_path.glob('*_confusion_matrix.png'))) == 4
