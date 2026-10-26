import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# W3D1: Linear Regression — Scikit-Learn
# ============================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. LOAD REAL HOUSE-PRICE DATASET
# ------------------------------------------------------------
housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("=== DATASET INFORMATION ===")
print("Shape:", df.shape)
print("\nColumns:")
print(list(df.columns))
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== TRAIN / TEST SPLIT ===")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 3. DEFINE MODELS
# ------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),

    # Scaling is useful for regularized models.
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=10000))
    ])
}


# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------
results = []
predictions = {}

for name, model in models.items():

    # Train model.
    model.fit(X_train, y_train)

    # Generate predictions.
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Calculate evaluation metrics.
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    print(f"\n=== {name} ===")
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("MAE :", round(mae, 4))
    print("R2  :", round(r2, 4))


# ------------------------------------------------------------
# 5. RESULTS COMPARISON
# ------------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)


# ------------------------------------------------------------
# 6. PRINT LINEAR REGRESSION COEFFICIENTS
# ------------------------------------------------------------
linear_model = models["Linear Regression"]

print("\n=== LINEAR REGRESSION COEFFICIENTS ===")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

print(coefficients)


# ------------------------------------------------------------
# 7. ACTUAL VS PREDICTED PLOT
# ------------------------------------------------------------
y_pred_linear = predictions["Linear Regression"]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 8. RESIDUAL PLOT
# ------------------------------------------------------------
residuals = y_test - y_pred_linear

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_linear, residuals, alpha=0.5)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted House Value")
plt.ylabel("Residual")
plt.title("Linear Regression: Residual Plot")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "residuals.png"),
    dpi=150
)

plt.close()


print("\n=== W3D1 COMPLETED SUCCESSFULLY ===")
print("Output files saved in:", OUTPUT_DIR)