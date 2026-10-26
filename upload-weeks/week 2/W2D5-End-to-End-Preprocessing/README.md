# W2D5: End-to-End Preprocessing Pipeline

## Project Overview

This project implements an end-to-end machine learning preprocessing pipeline using the Titanic dataset.

The pipeline applies the concepts learned during Week 2:

* Exploratory Data Analysis (EDA)
* Missing value handling
* Categorical encoding
* Feature scaling
* ML-ready feature generation
* Automated testing

## Objective

The objective is to transform raw Titanic data into a clean, numerical and scaled dataset that can be directly used for machine learning models.

## Dataset

The Titanic dataset contains passenger information such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses
* Number of parents/children
* Fare
* Embarkation port
* Survival status

The target variable is:

`Survived`

where:

* `0` = Did not survive
* `1` = Survived

## Preprocessing Workflow

```text
Raw Titanic Dataset
        ↓
Data Inspection
        ↓
EDA
        ↓
Remove Unnecessary Columns
        ↓
Separate Features and Target
        ↓
Handle Missing Values
        ↓
One-Hot Encode Categorical Features
        ↓
Standardize Numerical Features
        ↓
Generate ML-Ready Dataset
        ↓
Export CSV
```

## Missing Value Handling

Numerical features use median imputation.

Categorical features use the most frequent value.

This prevents missing values from causing problems during model training.

## Encoding

Categorical variables are converted into numerical features using One-Hot Encoding.

`handle_unknown="ignore"` is used so that unseen categories do not cause errors during transformation.

## Feature Scaling

Numerical features are standardized using `StandardScaler`.

This puts numerical features on a comparable scale and is useful for many machine learning algorithms.

## Columns Removed

The following columns were removed because they are not directly useful for the preprocessing model:

* `PassengerId`
* `Name`
* `Ticket`
* `Cabin`

## Output

The processed dataset is exported to:

```text
outputs/processed/titanic_ml_ready.csv
```

EDA visualisations are saved in:

```text
outputs/figures/
```

## Testing

Automated tests are implemented using `pytest`.

The tests verify:

* Input dataset exists
* Dataset is not empty
* Target column exists
* Processed dataset exists
* Processed dataset contains no missing values
* Processed dataset is not empty
* Target values are valid

Run tests using:

```bash
pytest
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pytest
* Git
* GitHub

## Project Structure

```text
W2D5-End-to-End-Preprocessing/
│
├── data/
│   └── titanic.csv
│
├── outputs/
│   ├── figures/
│   │   ├── age_distribution.png
│   │   └── survival_distribution.png
│   │
│   └── processed/
│       └── titanic_ml_ready.csv
│
├── src/
│   └── preprocessing.py
│
├── tests/
│   └── test_preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Result

The raw Titanic dataset was successfully transformed into a clean ML-ready dataset through a reproducible preprocessing pipeline.