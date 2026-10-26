import numpy as np
from src.logistic_classification import binary_model

def test_binary_predict_probability():
    X=np.array([[0.],[1.],[2.],[3.],[4.],[5.]])
    y=np.array([0,0,0,1,1,1])
    model=binary_model().fit(X,y)
    assert np.allclose(model.predict_proba(X).sum(axis=1),1)
