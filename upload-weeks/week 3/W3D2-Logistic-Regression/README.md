# W3D2 — Logistic Regression & Classification

Binary breast-cancer classification: coefficients, intercept, accuracy, precision, recall, F1, confusion matrix, ROC-AUC. Multiclass Iris comparison: One-vs-Rest versus multinomial (softmax) logistic regression.

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python -m src.logistic_classification
```

Evidence is generated in `outputs/`. Metrics should be interpreted in context; the breast-cancer dataset is educational, not a clinical tool.
