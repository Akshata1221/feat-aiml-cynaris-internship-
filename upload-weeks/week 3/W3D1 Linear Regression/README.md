# W3D1: Linear Regression — Scikit-Learn

## Objective

Implement and compare **Linear Regression, Ridge Regression, and Lasso Regression** using a real house-price dataset.

## Dataset

The **California Housing dataset** from scikit-learn was used.

* **Samples:** 20,640
* **Features:** 8
* **Target:** `MedHouseVal`

## Models

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

## Evaluation Metrics

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

## Results

| Model             |    MSE |   RMSE |    MAE |     R² |
| ----------------- | -----: | -----: | -----: | -----: |
| Linear Regression | 0.5559 | 0.7456 | 0.5332 | 0.5758 |
| Ridge             | 0.5559 | 0.7456 | 0.5332 | 0.5758 |
| Lasso             | 0.5545 | 0.7446 | 0.5331 | 0.5769 |

**Lasso achieved the best overall result** on the test set, with the lowest MSE, RMSE, and MAE and the highest R² score.

## Visualizations

The project generates the following visualizations:

* `output/actual_vs_predicted.png`
* `output/residuals.png`

## Output

The model comparison results are also saved as:

* `output/model_comparison.csv`

## How to Run

```bash
python linear_regression.py
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Git Commit

```text
feat: linear regression — train, evaluate, compare
```


## How to Run

```bash
python linear_regression.py
