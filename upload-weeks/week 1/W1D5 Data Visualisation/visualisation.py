import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/indian_student_performance.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("=== DATASET INFORMATION ===")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# Identify columns
# -----------------------------
numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

print("\nNumeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)

# -----------------------------
# 1. Numeric distributions
# -----------------------------
for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    filename = f"{column}_distribution.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# -----------------------------
# 2. Correlation heatmap
# -----------------------------
if len(numeric_columns) >= 2:
    plt.figure(figsize=(10, 7))

    correlation = df[numeric_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    )
    plt.close()

# -----------------------------
# 3. Top-10 categorical values
# -----------------------------
for column in categorical_columns:
    counts = df[column].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=counts.values,
        y=counts.index
    )

    plt.title(f"Top 10 Categories - {column}")
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.tight_layout()

    filename = f"{column}_top10_categories.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# -----------------------------
# 4. Boxplots for numeric data
# -----------------------------
for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column].dropna())

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.tight_layout()

    filename = f"{column}_boxplot.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

print("\n=== VISUALISATION COMPLETED ===")
# -----------------------------
# 5. Trend analysis
# -----------------------------
if "Study_Hours" in df.columns:
    score_columns = [
        column for column in
        ["Math_Score", "Science_Score", "English_Score"]
        if column in df.columns
    ]

    if score_columns:
        trend_data = df.groupby("Study_Hours")[score_columns].mean()

        plt.figure(figsize=(10, 6))

        for column in score_columns:
            plt.plot(
                trend_data.index,
                trend_data[column],
                marker="o",
                label=column
            )

        plt.title("Average Scores by Study Hours")
        plt.xlabel("Study Hours")
        plt.ylabel("Average Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "study_hours_score_trend.png")
        )
        plt.close()

        print("5. Study-hours score trend")
print("Generated visualizations:")
print("1. Numeric distribution plots")
print("2. Correlation heatmap")
print("3. Top-10 categorical plots")
print("4. Numeric boxplots")
print("\nOutput directory:", OUTPUT_DIR)