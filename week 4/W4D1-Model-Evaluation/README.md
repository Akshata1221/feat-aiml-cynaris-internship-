# W4D1 — Model Evaluation Metrics: Precision, Recall & AUC

Cynaris AI/ML 3M practical: evaluate Logistic Regression and Decision Tree on the **scikit-learn breast cancer dataset** (educational data, not a clinical tool). Uses stratified train/test split, 5-fold stratified cross-validation, ROC AUC, confusion matrices, learning curves, and MLflow tracking. The approved stack includes MLflow and MLOps; CrewAI, LangGraph and Ragas are not necessary for this classical evaluation task.

## Windows CMD setup

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python src\evaluate_models.py
mlflow ui --backend-store-uri mlruns
```

MLflow UI: http://127.0.0.1:5000. The script creates `outputs/evaluation_metrics.csv`, `outputs/run_summary.json`, confusion matrix PNGs, ROC PNGs and learning curve PNGs. **Run the script to generate authentic evidence**. MLflow run data stays local and is excluded from Git.

## Design decisions

- Stratification maintains class balance in each split/fold.
- Scaling is inside the Logistic Regression pipeline to avoid CV leakage.
- 5-fold CV uses training data only; held-out test data is evaluated once.
- AUC uses predicted probabilities rather than hard labels.
- Precision measures how many predicted positives are correct; recall measures how many actual positives were found. In this dataset positive class 1 means **benign**, so clinical interpretation of recall requires careful label choice.
- Learning curves compare training and validation accuracy to help identify overfitting (large persistent gap) or underfitting (both low). These are diagnostic indicators, not definitive proof.

## Deliverables

- `src/evaluate_models.py`: reproducible pipeline
- `tests/test_evaluation.py`: automated tests
- `outputs/`: generated evidence after execution
- `docs/self_review.md`: checklist
- `docs/viva_answers.md`: sample viva responses
- `docs/git_steps.md`: Git and PR instructions
- `docs/cia_interactions.md`: prompts for two actual CIA mentor interactions

## Submission notes

Run tests, inspect charts and metrics, record your actual CIA review and fixes, then make at least two descriptive commits and open a PR. Do not claim a PR or CIA interaction happened until you complete it.
