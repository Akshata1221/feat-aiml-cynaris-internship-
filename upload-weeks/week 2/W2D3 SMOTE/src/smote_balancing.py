"""
W2D3: Handling Imbalanced Data - SMOTE

This script:
1. Loads the student performance dataset.
2. Handles missing numerical values.
3. Checks duplicate records.
4. Encodes categorical input features.
5. Encodes the target variable.
6. Splits the dataset into training and testing sets.
7. Applies SMOTE only to the training data.
8. Compares class distributions before and after SMOTE.
9. Saves balanced data and visualizations.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/indian_student_performance.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("W2D3: HANDLING IMBALANCED DATA - SMOTE")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n=== LOADING DATASET ===")

df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n=== MISSING VALUES BEFORE IMPUTATION ===")

print(df.isnull().sum())


# ============================================================
# 3. HANDLE MISSING NUMERICAL VALUES
# ============================================================

numeric_columns = [
    "Study_Hours",
    "Attendance_Percent",
    "Math_Score",
    "Science_Score",
    "English_Score"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


print("\n=== MISSING VALUES AFTER IMPUTATION ===")

print(df[numeric_columns].isnull().sum())


# ============================================================
# 4. DUPLICATE CHECK
# ============================================================

print("\n=== DUPLICATE CHECK ===")

rows_before = len(df)

df = df.drop_duplicates()

rows_after = len(df)

print(f"Rows before: {rows_before}")
print(f"Rows after: {rows_after}")
print(f"Duplicates removed: {rows_before - rows_after}")


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

target_column = "Performance_Level"

print("\n=== TARGET CLASS DISTRIBUTION BEFORE SMOTE ===")

class_distribution = df[target_column].value_counts()

print(class_distribution)


# ============================================================
# 6. PLOT CLASS DISTRIBUTION BEFORE SMOTE
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target_column,
    order=df[target_column].value_counts().index
)

plt.title("Class Distribution Before SMOTE")
plt.xlabel("Performance Level")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution_before.png"),
    dpi=300
)

plt.close()


# ============================================================
# 7. ENCODE CATEGORICAL INPUT FEATURES
# ============================================================

print("\n=== ENCODING CATEGORICAL FEATURES ===")

# Gender
gender_encoder = LabelEncoder()

df["Gender_Encoded"] = gender_encoder.fit_transform(
    df["Gender"]
)

print("\nGender classes:")
print(gender_encoder.classes_)


# City
city_encoder = LabelEncoder()

df["City_Encoded"] = city_encoder.fit_transform(
    df["City"]
)

print("\nCity classes:")
print(city_encoder.classes_)


# ============================================================
# 8. ENCODE TARGET
# ============================================================

target_encoder = LabelEncoder()

df["Performance_Level_Encoded"] = target_encoder.fit_transform(
    df[target_column]
)

print("\nTarget classes:")
print(target_encoder.classes_)


# ============================================================
# 9. CREATE FEATURES AND TARGET
# ============================================================

feature_columns = [
    "Gender_Encoded",
    "City_Encoded",
    "Study_Hours",
    "Attendance_Percent",
    "Math_Score",
    "Science_Score",
    "English_Score"
]

X = df[feature_columns]

y = df["Performance_Level_Encoded"]


print("\n=== FEATURE MATRIX ===")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n=== TRAIN / TEST SPLIT ===")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 11. CLASS DISTRIBUTION BEFORE SMOTE
# ============================================================

print("\n=== TRAINING CLASS DISTRIBUTION BEFORE SMOTE ===")

before_smote = pd.Series(y_train).value_counts().sort_index()

before_smote_named = before_smote.copy()

before_smote_named.index = target_encoder.inverse_transform(
    before_smote_named.index
)

print(before_smote_named)


# ============================================================
# 12. APPLY SMOTE
# ============================================================

print("\n=== APPLYING SMOTE ===")

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# ============================================================
# 13. CLASS DISTRIBUTION AFTER SMOTE
# ============================================================

print("\n=== TRAINING CLASS DISTRIBUTION AFTER SMOTE ===")

after_smote = pd.Series(y_train_smote).value_counts().sort_index()

after_smote_named = after_smote.copy()

after_smote_named.index = target_encoder.inverse_transform(
    after_smote_named.index
)

print(after_smote_named)


# ============================================================
# 14. SMOTE SUMMARY
# ============================================================

original_training_samples = len(X_train)

balanced_training_samples = len(X_train_smote)

synthetic_samples = (
    balanced_training_samples -
    original_training_samples
)

print("\n=== SMOTE SUMMARY ===")

print(
    f"Original training samples: "
    f"{original_training_samples}"
)

print(
    f"Balanced training samples: "
    f"{balanced_training_samples}"
)

print(
    f"Synthetic samples created: "
    f"{synthetic_samples}"
)


# ============================================================
# 15. SAVE CLASS DISTRIBUTION
# ============================================================

distribution_df = pd.DataFrame({
    "Class": target_encoder.classes_,
    "Before_SMOTE": [
        before_smote.get(i, 0)
        for i in range(len(target_encoder.classes_))
    ],
    "After_SMOTE": [
        after_smote.get(i, 0)
        for i in range(len(target_encoder.classes_))
    ]
})

distribution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution.csv"
    ),
    index=False
)


# ============================================================
# 16. PLOT CLASS DISTRIBUTION AFTER SMOTE
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=distribution_df.melt(
        id_vars="Class",
        value_vars=[
            "Before_SMOTE",
            "After_SMOTE"
        ],
        var_name="Dataset",
        value_name="Count"
    ),
    x="Class",
    y="Count",
    hue="Dataset"
)

plt.title("Class Distribution Before and After SMOTE")
plt.xlabel("Performance Level")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution_after.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. SAVE BALANCED DATASET
# ============================================================

balanced_df = pd.DataFrame(
    X_train_smote,
    columns=feature_columns
)

balanced_df["Performance_Level_Encoded"] = y_train_smote

balanced_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "smote_balanced_dataset.csv"
    ),
    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("W2D3 SMOTE SUMMARY")
print("=" * 60)

print("\nTechnique:")
print("SMOTE (Synthetic Minority Over-sampling Technique)")

print("\nOriginal training distribution:")
print(before_smote_named)

print("\nBalanced training distribution:")
print(after_smote_named)

print("\nOutput files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(filename)

print("\n=== W2D3 COMPLETED SUCCESSFULLY ===")
print(f"Output directory: {OUTPUT_DIR}")