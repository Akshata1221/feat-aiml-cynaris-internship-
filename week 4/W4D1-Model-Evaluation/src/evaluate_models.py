"""Reproducible binary classification evaluation with stratified CV and MLflow."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SEED = 42


def make_models():
    return {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        'DecisionTree': DecisionTreeClassifier(max_depth=4, random_state=SEED),
    }


def metric_values(y_true, y_pred, y_score):
    """Scores use positive class 1; zero_division avoids undefined precision."""
    return dict(accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_true, y_score))


def run():
    import mlflow
    OUT.mkdir(exist_ok=True)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('W4D1_Model_Evaluation')
    rows = []
    for name, model in make_models().items():
        # CV uses training data only to avoid test-set leakage.
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring={'precision': 'precision', 'recall': 'recall',
                                         'f1': 'f1', 'auc': 'roc_auc'}, n_jobs=1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics = metric_values(y_test, pred, prob)
        metrics.update({f'cv_{key}_mean': float(np.mean(scores[f'test_{key}']))
                        for key in ('precision', 'recall', 'f1', 'auc')})
        metrics['model'] = name
        rows.append(metrics)
        with mlflow.start_run(run_name=name):
            mlflow.log_param('model', name)
            mlflow.log_param('cv_folds', 5)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if k != 'model'})
            ConfusionMatrixDisplay.from_predictions(y_test, pred)
            plt.title(f'{name} confusion matrix')
            plt.tight_layout()
            figpath = OUT / f'{name}_confusion_matrix.png'
            plt.savefig(figpath, dpi=150)
            plt.close()
            mlflow.log_artifact(str(figpath))
            RocCurveDisplay.from_predictions(y_test, prob)
            plt.title(f'{name} ROC curve')
            plt.tight_layout()
            figpath = OUT / f'{name}_roc_curve.png'
            plt.savefig(figpath, dpi=150)
            plt.close()
            mlflow.log_artifact(str(figpath))
            sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=cv, train_sizes=np.linspace(.2, 1., 5),
                scoring='accuracy', n_jobs=1)
            plt.figure()
            plt.plot(sizes, train_scores.mean(axis=1), 'o-', label='Training accuracy')
            plt.plot(sizes, val_scores.mean(axis=1), 'o-', label='CV accuracy')
            plt.xlabel('Training examples')
            plt.ylabel('Accuracy')
            plt.title(f'{name} learning curve')
            plt.legend()
            plt.tight_layout()
            figpath = OUT / f'{name}_learning_curve.png'
            plt.savefig(figpath, dpi=150)
            plt.close()
            mlflow.log_artifact(str(figpath))
    df = pd.DataFrame(rows).set_index('model')
    df.to_csv(OUT / 'evaluation_metrics.csv')
    summary = {'dataset': 'sklearn breast cancer (educational use only)',
               'train_samples': len(y_train), 'test_samples': len(y_test),
               'positive_class': 'benign (label 1)',
               'best_by_test_auc': str(df['roc_auc'].idxmax())}
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(df.round(4).to_string())
    print('\nSaved evidence to:', OUT)


if __name__ == '__main__':
    run()
