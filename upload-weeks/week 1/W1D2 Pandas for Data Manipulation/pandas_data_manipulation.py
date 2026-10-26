import os
import pandas as pd


# ============================================================
# W1D2: Pandas for Data Manipulation
# ============================================================

INPUT_FILE = "data/india_covid_case_data.csv"
OUTPUT_DIR = "output"


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("=== DATASET INFORMATION ===")
print("Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))


# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------
# Convert Date_YMD from string to datetime for easier analysis.
df["Date_YMD"] = pd.to_datetime(df["Date_YMD"])

# Remove duplicate rows.
df = df.drop_duplicates()

print("\n=== AFTER CLEANING ===")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 3. FILTER OPERATION
# ------------------------------------------------------------
# Filter days where daily confirmed cases were greater than 10.
filtered_df = df[df["Daily Confirmed"] > 10]

print("\n=== FILTER: Daily Confirmed > 10 ===")
print(filtered_df.head())


# ------------------------------------------------------------
# 4. GROUPBY OPERATION
# ------------------------------------------------------------
# Group records by month and calculate total confirmed cases.
df["Month"] = df["Date_YMD"].dt.to_period("M").astype(str)

monthly_cases = (
    df.groupby("Month")["Daily Confirmed"]
    .sum()
    .reset_index(name="Monthly Confirmed")
)

print("\n=== GROUPBY: Monthly Confirmed Cases ===")
print(monthly_cases)


# ------------------------------------------------------------
# 5. MERGE OPERATION
# ------------------------------------------------------------
# Create a small reference DataFrame containing month categories.
month_reference = pd.DataFrame({
    "Month": monthly_cases["Month"],
    "Period": [
        "Early Pandemic" if month < "2020-04"
        else "Pandemic Period"
        for month in monthly_cases["Month"]
    ]
})

# Merge monthly statistics with the reference DataFrame.
merged_df = pd.merge(
    monthly_cases,
    month_reference,
    on="Month",
    how="left"
)

print("\n=== MERGE ===")
print(merged_df.head(10))


# ------------------------------------------------------------
# 6. PIVOT TABLE OPERATION
# ------------------------------------------------------------
# Create a pivot table showing monthly confirmed cases
# according to the pandemic period.
pivot_df = pd.pivot_table(
    merged_df,
    values="Monthly Confirmed",
    index="Month",
    columns="Period",
    aggfunc="sum",
    fill_value=0
)

print("\n=== PIVOT TABLE ===")
print(pivot_df)


# ------------------------------------------------------------
# 7. EXPORT CLEANED DATA
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_output = os.path.join(
    OUTPUT_DIR,
    "cleaned_india_covid_cases.csv"
)

parquet_output = os.path.join(
    OUTPUT_DIR,
    "cleaned_india_covid_cases.parquet"
)

# Export cleaned DataFrame to CSV.
df.to_csv(csv_output, index=False)

# Export cleaned DataFrame to Parquet.
df.to_parquet(parquet_output, index=False)


# ------------------------------------------------------------
# 8. COMPARE FILE SIZES
# ------------------------------------------------------------
csv_size = os.path.getsize(csv_output)
parquet_size = os.path.getsize(parquet_output)

print("\n=== FILE SIZE COMPARISON ===")
print(f"CSV size:     {csv_size:,} bytes")
print(f"Parquet size: {parquet_size:,} bytes")

if parquet_size < csv_size:
    print("Parquet file is smaller than CSV.")
elif csv_size < parquet_size:
    print("CSV file is smaller for this small dataset.")
else:
    print("Both files have the same size.")


print("\n=== W1D2 COMPLETED SUCCESSFULLY ===")