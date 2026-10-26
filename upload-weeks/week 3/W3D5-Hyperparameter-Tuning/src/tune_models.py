"""Reproducible SVM/KNN hyperparameter tuning with MLflow tracking."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def make_searches(seed=42, cv_folds=3, n_iter=8):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    svm = Pipeline([('scale', StandardScaler()), ('model', SVC())])
    knn = Pipeline([('scale', StandardScaler()), ('model', KNeighborsClassifier())])
    svm_space = {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf'], 'model__gamma': ['scale', 'auto']}
    knn_space = {'model__n_neighbors': [3, 5, 7, 9, 11], 'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]}
    return {
        'SVM_GridSearch': GridSearchCV(svm, svm_space, cv=cv, scoring='f1', n_jobs=-1),
        'SVM_RandomSearch': RandomizedSearchCV(svm, svm_space, n_iter=min(n_iter, 12), cv=cv, scoring='f1', random_state=seed, n_jobs=-1),
        'KNN_GridSearch': GridSearchCV(knn, knn_space, cv=cv, scoring='f1', n_jobs=-1),
        'KNN_RandomSearch': RandomizedSearchCV(knn, knn_space, n_iter=min(n_iter, 20), cv=cv, scoring='f1', random_state=seed, n_jobs=-1),
    }


def run(output_dir=ROOT / 'outputs', seed=42, cv_folds=3, n_iter=8, use_mlflow=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, stratify=data.target, random_state=seed)
    mlflow = None
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
        mlflow.set_experiment('W3D5_Hyperparameter_Tuning')
    rows = []
    for name, search in make_searches(seed, cv_folds, n_iter).items():
        started = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        pred = search.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
        }
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['malignant', 'benign']).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(output_dir / f'{name}_confusion_matrix.png', dpi=140)
        plt.close(fig)
        row = {'model': name, 'cv_f1': search.best_score_, **metrics, 'duration_seconds': round(elapsed, 3), 'candidates': len(search.cv_results_['params']), 'best_params': json.dumps(search.best_params_, sort_keys=True)}
        rows.append(row)
        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({'model': name, 'seed': seed, 'cv_folds': cv_folds, 'search_candidates': row['candidates'], **search.best_params_})
                mlflow.log_metrics({**metrics, 'cv_f1': search.best_score_, 'duration_seconds': elapsed})
                mlflow.log_artifact(str(output_dir / f'{name}_confusion_matrix.png'))
    result = pd.DataFrame(rows).sort_values(['f1', 'cv_f1'], ascending=False)
    result.to_csv(output_dir / 'model_comparison.csv', index=False)
    (output_dir / 'best_model.json').write_text(json.dumps(result.iloc[0].to_dict(), indent=2), encoding='utf-8')
    print(result.to_string(index=False))
    print(f'\nSaved evidence to {output_dir}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=8)
    args = parser.parse_args()
    run(cv_folds=args.cv_folds, n_iter=args.n_iter, use_mlflow=not args.no_mlflow)
