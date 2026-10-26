# W3D5 — Hyperparameter Tuning: GridSearch & RandomSearch

A reproducible comparison of **SVM** and **KNN** classifiers tuned with **GridSearchCV** and **RandomizedSearchCV**, with MLflow experiment tracking.

## Dataset and design
- Built-in scikit-learn Wisconsin Breast Cancer dataset (569 samples, 30 numeric features).
- Stratified 80/20 train/test split, `random_state=42`.
- `StandardScaler` is inside each model pipeline to avoid cross-validation leakage.
- Stratified 3-fold cross-validation; F1 score used to choose hyperparameters.
- Four experiments: SVM Grid, SVM Random, KNN Grid, KNN Random.
- Final holdout test scores are used only for evaluation. This is an educational dataset, not a medical diagnostic system.

## Run in Windows Command Prompt
```bat
cd W3D5-Hyperparameter-Tuning
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python src\tune_models.py
mlflow ui --backend-store-uri mlruns
```
Open `http://127.0.0.1:5000` for experiment tracking. If the MLflow UI is not needed, use `python src\tune_models.py --no-mlflow`.

## Outputs
Running the script generates `outputs/model_comparison.csv`, `outputs/best_model.json`, and four confusion-matrix PNG files. Runtime and best hyperparameters are saved in the CSV. Results depend on environment and package versions; don't report metrics before running.

## Search-space design
- SVM: C = 0.1, 1, 10; kernel = linear/rbf; gamma = scale/auto (12 combinations).
- KNN: neighbors = 3, 5, 7, 9, 11; weights = uniform/distance; p = 1/2 (20 combinations).
- RandomizedSearchCV evaluates 8 sampled configurations per model by default, reducing the search budget.

## Approved stack alignment
MLflow is used for experiment tracking, while the preprocessing and model training are implemented in scikit-learn as required by this lesson. CrewAI, LangGraph, and Ragas are not forced into a classical hyperparameter-tuning task where they add no value. Reproducible pipelines, tests, and experiment artifacts support MLOps practices.

## Deliverables checklist
See `docs/self_review.md`, `docs/viva_answers.md`, and `docs/git_steps.md`.
