"""
W2D5: End-to-End Preprocessing Pipeline

Pipeline:
1. Load Titanic dataset
2. Perform basic EDA
3. Handle missing values
4. Encode categorical features
5. Scale numerical features
6. Export ML-ready features
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

DATA_PATH = "data/titanic.csv"
OUTPUT_PATH = "outputs/processed/titanic_ml_ready.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TITANIC DATASET - INITIAL INSPECTION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# 2. Basic EDA
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())

print("\nSurvival distribution:")
print(df["Survived"].value_counts())


# ---------------------------------------------------------
# 3. EDA Visualisations
# ---------------------------------------------------------

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/age_distribution.png")
plt.close()


# Survival distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/survival_distribution.png")
plt.close()


# ---------------------------------------------------------
# 4. Separate Features and Target
# ---------------------------------------------------------

TARGET = "Survived"

# Remove columns that are not useful for the ML model
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]

X = df.drop(columns=[TARGET] + DROP_COLUMNS)
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(TARGET)


# ---------------------------------------------------------
# 5. Identify Numerical and Categorical Columns
# ---------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ---------------------------------------------------------
# 6. Preprocessing Pipelines
# ---------------------------------------------------------

# Numerical preprocessing:
# Missing values → median
# Scaling → StandardScaler

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing:
# Missing values → most frequent value
# Encoding → One-Hot Encoding

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ---------------------------------------------------------
# 7. Combine Preprocessing Steps
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ---------------------------------------------------------
# 8. Transform Dataset
# ---------------------------------------------------------

X_processed = preprocessor.fit_transform(X)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Add target column
processed_df[TARGET] = y.values


# ---------------------------------------------------------
# 9. Export ML-Ready Dataset
# ---------------------------------------------------------

processed_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nProcessed dataset shape:")
print(processed_df.shape)

print("\nProcessed columns:")
print(processed_df.columns.tolist())

print("\nRemaining missing values:")
print(processed_df.isnull().sum().sum())

print(f"\nML-ready dataset saved to: {OUTPUT_PATH}")