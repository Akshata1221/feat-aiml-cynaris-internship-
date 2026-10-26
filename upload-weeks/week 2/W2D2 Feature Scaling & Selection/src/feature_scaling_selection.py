import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/indian_student_performance.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("W2D2: FEATURE SCALING & SELECTION")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET INFORMATION ===")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CHECK AND HANDLE MISSING VALUES
# ============================================================

print("\n=== MISSING VALUES BEFORE IMPUTATION ===")
print(df.isnull().sum())

numeric_columns = [
    "Study_Hours",
    "Attendance_Percent",
    "Math_Score",
    "Science_Score",
    "English_Score"
]

# Fill missing numeric values using median
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\n=== MISSING VALUES AFTER IMPUTATION ===")
print(df[numeric_columns].isnull().sum())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\n=== DUPLICATE CHECK ===")
print("Rows before:", before_duplicates)
print("Rows after:", after_duplicates)
print("Duplicates removed:", before_duplicates - after_duplicates)


# ============================================================
# 4. LABEL ENCODER
# ============================================================

print("\n=== LABEL ENCODER ===")

label_encoder = LabelEncoder()

df["Gender_LabelEncoded"] = label_encoder.fit_transform(
    df["Gender"]
)

print("Gender classes:")
print(label_encoder.classes_)

print("\nEncoded Gender:")
print(
    df[
        ["Gender", "Gender_LabelEncoded"]
    ].head()
)


# ============================================================
# 5. ONE HOT ENCODER
# ============================================================

print("\n=== ONE HOT ENCODER ===")

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

city_encoded = one_hot_encoder.fit_transform(
    df[["City"]]
)

city_columns = one_hot_encoder.get_feature_names_out(
    ["City"]
)

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns,
    index=df.index
)

print("One-hot encoded City columns:")
print(city_columns)

print("\nSample:")
print(city_encoded_df.head())


# ============================================================
# 6. ORDINAL ENCODER
# ============================================================

print("\n=== ORDINAL ENCODER ===")

# Explicit order is important because Performance_Level
# has a meaningful ranking.
performance_order = [
    ["Needs Improvement", "Average", "Good", "Excellent"]
]

ordinal_encoder = OrdinalEncoder(
    categories=performance_order,
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df["Performance_OrdinalEncoded"] = ordinal_encoder.fit_transform(
    df[["Performance_Level"]]
).ravel()

print(
    df[
        [
            "Performance_Level",
            "Performance_OrdinalEncoded"
        ]
    ].head()
)


# ============================================================
# 7. SAVE ENCODED FEATURES
# ============================================================

encoded_output = pd.concat(
    [
        df,
        city_encoded_df
    ],
    axis=1
)

encoded_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "encoded_features.csv"
    ),
    index=False
)


# ============================================================
# 8. NUMERIC FEATURES
# ============================================================

X = df[numeric_columns].copy()

print("\n=== NUMERIC FEATURES ===")
print(X.head())


# ============================================================
# 9. SCALING
# ============================================================

print("\n=== STANDARD SCALER ===")

standard_scaler = StandardScaler()

X_standard = standard_scaler.fit_transform(X)

standard_df = pd.DataFrame(
    X_standard,
    columns=numeric_columns
)

print(standard_df.head())

standard_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "standard_scaled_features.csv"
    ),
    index=False
)


print("\n=== MINMAX SCALER ===")

minmax_scaler = MinMaxScaler()

X_minmax = minmax_scaler.fit_transform(X)

minmax_df = pd.DataFrame(
    X_minmax,
    columns=numeric_columns
)

print(minmax_df.head())

minmax_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "minmax_scaled_features.csv"
    ),
    index=False
)


print("\n=== ROBUST SCALER ===")

robust_scaler = RobustScaler()

X_robust = robust_scaler.fit_transform(X)

robust_df = pd.DataFrame(
    X_robust,
    columns=numeric_columns
)

print(robust_df.head())

robust_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "robust_scaled_features.csv"
    ),
    index=False
)


# ============================================================
# 10. PLOTS BEFORE SCALING
# ============================================================

print("\n=== CREATING BEFORE-SCALING PLOTS ===")

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        X[column],
        kde=True
    )

    plt.title(
        f"{column} - Before Scaling"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{column}_before_scaling.png"
        )
    )

    plt.close()


# ============================================================
# 11. PLOTS AFTER SCALING
# ============================================================

print("\n=== CREATING AFTER-SCALING PLOTS ===")

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        standard_df[column],
        kde=True
    )

    plt.title(
        f"{column} - After Standard Scaling"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{column}_after_standard_scaling.png"
        )
    )

    plt.close()


# ============================================================
# 12. COMPARISON OF SCALERS
# ============================================================

print("\n=== CREATING SCALER COMPARISON PLOTS ===")

for column in numeric_columns:

    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        standard_df[column],
        label="StandardScaler"
    )

    sns.kdeplot(
        minmax_df[column],
        label="MinMaxScaler"
    )

    sns.kdeplot(
        robust_df[column],
        label="RobustScaler"
    )

    plt.title(
        f"Scaling Comparison - {column}"
    )

    plt.xlabel("Scaled Value")
    plt.ylabel("Density")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{column}_scaler_comparison.png"
        )
    )

    plt.close()


# ============================================================
# 13. SELECT K BEST
# ============================================================

print("\n=== SELECT K BEST: TOP 5 FEATURES ===")

# Encode the target variable
target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    df["Performance_Level"]
)

# Use imputed numeric features
X_features = df[numeric_columns].copy()

# Select top 5 features
selector = SelectKBest(
    score_func=f_classif,
    k=5
)

X_selected = selector.fit_transform(
    X_features,
    y
)

selected_mask = selector.get_support()

selected_features = [
    feature
    for feature, selected
    in zip(numeric_columns, selected_mask)
    if selected
]

scores = selector.scores_

feature_scores = pd.DataFrame(
    {
        "Feature": numeric_columns,
        "Score": scores
    }
).sort_values(
    by="Score",
    ascending=False
)

print("\nFeature scores:")
print(feature_scores)

print("\nTop 5 selected features:")
print(selected_features)


# ============================================================
# 14. SAVE SELECTK BEST RESULTS
# ============================================================

feature_scores.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "select_k_best_feature_scores.csv"
    ),
    index=False
)

top_5_df = pd.DataFrame(
    {
        "Top_5_Features": selected_features
    }
)

top_5_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_5_features.csv"
    ),
    index=False
)


# ============================================================
# 15. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("W2D2 FEATURE ENGINEERING SUMMARY")
print("=" * 60)

print("\nEncoders applied:")
print("1. LabelEncoder - Gender")
print("2. OneHotEncoder - City")
print("3. OrdinalEncoder - Performance_Level")

print("\nScalers applied:")
print("1. StandardScaler")
print("2. MinMaxScaler")
print("3. RobustScaler")

print("\nFeature selection:")
print("SelectKBest selected:")
print(selected_features)

print("\n=== OUTPUT FILES ===")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(filename)

print("\n=== W2D2 COMPLETED SUCCESSFULLY ===")
print("Output directory:", OUTPUT_DIR)