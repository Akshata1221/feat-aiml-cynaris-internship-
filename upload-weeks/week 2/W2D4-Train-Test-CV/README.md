# W2D4 — Train/Test Split & Cross-Validation

Stratified holdout and 5-fold CV; StandardScaler, MinMaxScaler, and RobustScaler are fit **inside** pipelines to prevent data leakage. Dataset: scikit-learn breast cancer dataset.

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python -m src.evaluate_split
```

Evidence: `outputs/scaler_comparison.csv`, `outputs/split_summary.json` after running.
