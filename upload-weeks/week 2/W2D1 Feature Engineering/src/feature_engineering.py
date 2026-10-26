import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from sklearn.feature_selection import SelectKBest, f_regression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/indian_student_performance.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=== W2D1 FEATURE ENGINEERING & ENCODING ===")


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
# 2. CHECK MISSING VALUES
# ============================================================

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates().copy()

print("\nShape after removing duplicates:", df.shape)


# ============================================================
# 4. HANDLE MISSING NUMERIC VALUES
# ============================================================

numeric_columns = [
    "Study_Hours",
    "Attendance_Percent",
    "Math_Score",
    "Science_Score",
    "English_Score",
]

# Use median because it is less affected by outliers
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\n=== MISSING VALUES AFTER IMPUTATION ===")
print(df[numeric_columns].isnull().sum())


# ============================================================
# 5. LABEL ENCODER
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
# 6. ONE HOT ENCODER
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
# 7. ORDINAL ENCODER
# ============================================================

print("\n=== ORDINAL ENCODER ===")

# Correct order based on performance level
performance_order = [
    [
        "Needs Improvement",
        "Average",
        "Good",
        "Excellent"
    ]
]

ordinal_encoder = OrdinalEncoder(
    categories=performance_order
)

df["Performance_OrdinalEncoded"] = (
    ordinal_encoder.fit_transform(
        df[["Performance_Level"]]
    ).ravel()
)

print(
    df[
        [
            "Performance_Level",
            "Performance_OrdinalEncoded"
        ]
    ].head()
)


# ============================================================
# 8. NUMERIC FEATURES
# ============================================================

print("\n=== NUMERIC FEATURES ===")

numeric_features = df[numeric_columns].copy()

print(numeric_features.head())


# ============================================================
# 9. STANDARD SCALER
# ============================================================

print("\n=== StandardScaler ===")

standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(
    numeric_features
)

standard_scaled_df = pd.DataFrame(
    standard_scaled,
    columns=numeric_columns,
    index=df.index
)

print(standard_scaled_df.head())


# ============================================================
# 10. MIN-MAX SCALER
# ============================================================

print("\n=== MinMaxScaler ===")

minmax_scaler = MinMaxScaler()

minmax_scaled = minmax_scaler.fit_transform(
    numeric_features
)

minmax_scaled_df = pd.DataFrame(
    minmax_scaled,
    columns=numeric_columns,
    index=df.index
)

print(minmax_scaled_df.head())


# ============================================================
# 11. ROBUST SCALER
# ============================================================

print("\n=== RobustScaler ===")

robust_scaler = RobustScaler()

robust_scaled = robust_scaler.fit_transform(
    numeric_features
)

robust_scaled_df = pd.DataFrame(
    robust_scaled,
    columns=numeric_columns,
    index=df.index
)

print(robust_scaled_df.head())


# ============================================================
# 12. SAVE ENCODED DATA
# ============================================================

encoded_data = pd.concat(
    [
        df,
        city_encoded_df
    ],
    axis=1
)

encoded_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "encoded_features.csv"
    ),
    index=False
)


# ============================================================
# 13. SAVE SCALED DATA
# ============================================================

standard_scaled_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "standard_scaled_features.csv"
    ),
    index=False
)

minmax_scaled_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "minmax_scaled_features.csv"
    ),
    index=False
)

robust_scaled_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "robust_scaled_features.csv"
    ),
    index=False
)


# ============================================================
# 14. PLOTS BEFORE SCALING
# ============================================================

print("\n=== CREATING BEFORE-SCALING PLOTS ===")

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        numeric_features[column],
        bins=15
    )

    plt.title(
        f"Before Scaling - {column}"
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
# 15. PLOTS AFTER STANDARD SCALING
# ============================================================

print("\n=== CREATING AFTER-SCALING PLOTS ===")

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        standard_scaled_df[column],
        bins=15
    )

    plt.title(
        f"After StandardScaler - {column}"
    )

    plt.xlabel(
        f"{column} (Standardized)"
    )

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
# 16. SELECT K BEST
# ============================================================

print("\n=== SELECT K BEST: TOP 5 FEATURES ===")

# Target variable
# Performance level is ordinal:
# Needs Improvement = 0
# Average = 1
# Good = 2
# Excellent = 3

y = df["Performance_OrdinalEncoded"]

# Features used for feature selection
X_features = numeric_features.copy()

# Select the five best numerical features
selector = SelectKBest(
    score_func=f_regression,
    k=5
)

X_selected = selector.fit_transform(
    X_features,
    y
)

selected_features = X_features.columns[
    selector.get_support()
]

scores = selector.scores_

feature_scores = pd.DataFrame(
    {
        "Feature": X_features.columns,
        "Score": scores
    }
)

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature scores:")
print(feature_scores)

print("\nTop 5 selected features:")
print(selected_features.tolist())


# ============================================================
# 17. SAVE FEATURE SELECTION RESULTS
# ============================================================

feature_scores.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "select_k_best_feature_scores.csv"
    ),
    index=False
)

selected_features_df = pd.DataFrame(
    {
        "Selected_Feature": selected_features
    }
)

selected_features_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_5_features.csv"
    ),
    index=False
)


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n=== FEATURE ENGINEERING SUMMARY ===")

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
print(selected_features.tolist())


# ============================================================
# 19. OUTPUT FILES
# ============================================================

print("\n=== OUTPUT FILES ===")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(filename)


print("\n=== W2D1 COMPLETED SUCCESSFULLY ===")
print("Output directory:", OUTPUT_DIR)