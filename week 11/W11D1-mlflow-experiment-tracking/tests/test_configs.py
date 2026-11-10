from mlflow_tracking.train import CONFIGS

def test_five_distinct_experiment_configs():
    assert len(CONFIGS) == 5
    assert len({tuple(sorted(c.items())) for c in CONFIGS}) == 5
