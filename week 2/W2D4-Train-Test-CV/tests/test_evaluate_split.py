from src.evaluate_split import build_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def test_pipeline_has_scaler():
    p=build_pipeline(StandardScaler())
    assert isinstance(p,Pipeline)
    assert isinstance(p.steps[0][1],StandardScaler)
