"""Compare holdout and stratified CV without leaking scaling statistics."""
from pathlib import Path
import json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE=Path(__file__).resolve().parents[1]

def build_pipeline(scaler):
    return make_pipeline(scaler, LogisticRegression(max_iter=2000, random_state=42))

def run():
    X,y=load_breast_cancer(return_X_y=True)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]
    for name, scaler in [('standard',StandardScaler()),('minmax',MinMaxScaler()),('robust',RobustScaler())]:
        model=build_pipeline(scaler)
        scores=cross_validate(model,Xtrain,ytrain,cv=cv,scoring=['accuracy','f1'])
        model.fit(Xtrain,ytrain)
        pred=model.predict(Xtest)
        rows.append({'scaler':name,'cv_accuracy_mean':float(scores['test_accuracy'].mean()),'cv_f1_mean':float(scores['test_f1'].mean()),'holdout_accuracy':float(accuracy_score(ytest,pred)),'holdout_f1':float(f1_score(ytest,pred))})
    out=BASE/'outputs';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'scaler_comparison.csv',index=False)
    (out/'split_summary.json').write_text(json.dumps({'train_size':len(ytrain),'test_size':len(ytest),'cv_folds':5},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':run()
