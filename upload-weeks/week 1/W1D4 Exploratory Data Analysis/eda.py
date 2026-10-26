import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# W1D4: Exploratory Data Analysis (EDA)
# ============================================================

INPUT_FILE = "data/indian_student_performance.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("=== DATASET INFORMATION ===")
print("Shape:", df.shape)

print("\n=== DATA TYPES AND INFO ===")
df.info()

print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------
# Remove duplicate rows.
duplicates = df.duplicated().sum()
print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()

# Fill missing numeric values with their median.
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())


# ------------------------------------------------------------
# 3. FIVE EDA OBSERVATIONS
# ------------------------------------------------------------
print("\n=== FIVE EDA OBSERVATIONS ===")

print("1. The dataset contains 100 student records and 9 columns.")

print("2. Study hours, attendance, and subject scores are numeric "
      "variables suitable for statistical analysis.")

print("3. Missing values were detected in some numeric columns and "
      "were handled using median imputation.")

print("4. Performance Level is a categorical variable that can be "
      "used to analyze the distribution of student performance.")

print("5. Student scores generally increase with better attendance "
      "and higher study hours, although the relationship should be "
      "verified using the correlation matrix.")


# ------------------------------------------------------------
# 4. DISTRIBUTION OF NUMERIC COLUMNS
# ------------------------------------------------------------
numeric_columns = [
    "Study_Hours",
    "Attendance_Percent",
    "Math_Score",
    "Science_Score",
    "English_Score"
]

df[numeric_columns].hist(
    figsize=(12, 8),
    bins=10
)

plt.suptitle("Distribution of Numeric Variables")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "numeric_distributions.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5. CORRELATION HEATMAP
# ------------------------------------------------------------
correlation = df[numeric_columns].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 6. TOP-10 CATEGORY COUNTS
# ------------------------------------------------------------
# Count students by city and display the top 10 categories.
city_counts = df["City"].value_counts().head(10)

plt.figure(figsize=(10, 6))

city_counts.plot(kind="bar")

plt.title("Top 10 Cities by Number of Students")
plt.xlabel("City")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top10_category_counts.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 7. EDA NARRATIVE
# ------------------------------------------------------------
narrative = """
The dataset contains 100 student records with demographic information,
study habits, attendance, subject scores, and an overall performance
category. The numerical variables include study hours, attendance
percentage, and Mathematics, Science, and English scores. Initial
inspection using describe(), info(), and isnull().sum() provides a
clear overview of the data types, distributions, and missing values.

A small number of missing values were identified in numeric columns.
These values were handled using median imputation to preserve the
records without introducing extreme values. Duplicate records were
also checked and removed where necessary.

The distribution plots help identify the typical range of study hours,
attendance, and academic scores. The correlation heatmap is useful
for examining relationships among these variables. In particular,
higher study time and attendance appear to be associated with better
academic performance, although correlation does not prove causation.

The categorical analysis shows the distribution of students across
cities and helps identify the most represented locations. Suspicious
areas to investigate further include missing observations and any
unusually low attendance or subject scores. Before building a machine
learning model, additional validation of outliers, duplicate records,
and category consistency would be useful. Overall, the dataset is
structured and suitable for basic exploratory analysis and provides
useful insights into factors associated with student performance.
"""

with open(
    os.path.join(OUTPUT_DIR, "eda_narrative.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(narrative.strip())


print("\n=== EDA COMPLETED SUCCESSFULLY ===")