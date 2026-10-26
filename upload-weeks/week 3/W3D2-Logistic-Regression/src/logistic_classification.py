"""Binary and multiclass logistic regression experiments."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

BASE=Path(__file__).resolve().parents[1]

def binary_model():
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))

def main():
    out=BASE/'outputs';out.mkdir(exist_ok=True)
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model=binary_model();model.fit(Xtr,ytr)
    pred=model.predict(Xte);prob=model.predict_proba(Xte)[:,1]
    metrics={'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'roc_auc':roc_auc_score(yte,prob)}
    print('Coefficients:',model.named_steps['logisticregression'].coef_)
    print('Intercept:',model.named_steps['logisticregression'].intercept_)
    print('Confusion matrix:',confusion_matrix(yte,pred))
    pd.DataFrame([metrics]).to_csv(out/'binary_metrics.csv',index=False)
    ConfusionMatrixDisplay.from_predictions(yte,pred,cmap='Blues');plt.tight_layout();plt.savefig(out/'confusion_matrix.png');plt.close()
    fpr,tpr,_=roc_curve(yte,prob)
    plt.plot(fpr,tpr,label=f'AUC={metrics["roc_auc"]:.3f}');plt.plot([0,1],[0,1],'--');plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');plt.legend();plt.tight_layout();plt.savefig(out/'roc_curve.png');plt.close()
    Xi,yi=load_iris(return_X_y=True)
    a,b,c,d=train_test_split(Xi,yi,test_size=.2,stratify=yi,random_state=42)
    # Current sklearn defaults to multinomial for 3+ classes; avoid deprecated multi_class parameter.
    variants={'one_vs_rest':OneVsRestClassifier(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))), 'softmax':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))}
    rows=[]
    for name,clf in variants.items():
        clf.fit(a,c);p=clf.predict(b)
        rows.append({'method':name,'accuracy':accuracy_score(d,p),'macro_f1':f1_score(d,p,average='macro')})
    pd.DataFrame(rows).to_csv(out/'multiclass_comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
