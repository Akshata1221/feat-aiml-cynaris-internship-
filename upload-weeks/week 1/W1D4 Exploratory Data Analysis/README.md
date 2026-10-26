# W1D4: Exploratory Data Analysis (EDA)

## Overview

This project implements Exploratory Data Analysis (EDA) on the **Indian Student Performance** dataset using Python, Pandas, Matplotlib, and Seaborn.

The purpose of EDA is to understand the structure, quality, distribution, and relationships within a dataset before applying machine learning models.

## Objective

The main objectives of this task are:

* Load a real-world CSV dataset using Pandas.
* Explore the dataset using `info()`, `describe()`, and `isnull().sum()`.
* Identify missing values and potential data-quality issues.
* Analyze the distribution of numerical variables.
* Study relationships between numerical features using correlation analysis.
* Analyze categorical variables using top-10 category counts.
* Document important observations from the dataset.
* Prepare a short EDA narrative describing the findings and potential issues.

## Dataset

**File:** `data/indian_student_performance.csv`

The dataset contains student-related performance information and is used to demonstrate practical exploratory data analysis techniques.

The dataset is loaded into a Pandas DataFrame and analyzed using both numerical and categorical features.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Project Structure

```text
W1D4 Exploratory Data Analysis/
│
├── data/
│   └── indian_student_performance.csv
│
├── output/
│   ├── numeric_distributions.png
│   ├── correlation_heatmap.png
│   ├── top10_category_counts.png
│   └── eda_narrative.txt
│
├── .gitignore
├── README.md
├── eda.py
└── requirements.txt
```

## EDA Operations Performed

### 1. Dataset Inspection

The following Pandas functions were used:

```python
df.info()
df.describe()
df.isnull().sum()
```

These operations help identify:

* Number of rows and columns
* Data types
* Numerical statistics
* Missing values
* Potential data-quality problems

### 2. Numerical Distributions

Distribution plots were generated for numerical columns to understand:

* Central tendency
* Spread of values
* Skewness
* Possible outliers
* Unusual distributions

The resulting visualization is saved as:

```text
output/numeric_distributions.png
```

### 3. Correlation Analysis

A correlation matrix was calculated for numerical variables and visualized using a heatmap.

The correlation heatmap helps identify:

* Positive relationships
* Negative relationships
* Weak relationships
* Strongly correlated features

Output:

```text
output/correlation_heatmap.png
```

### 4. Top-10 Category Counts

Categorical variables were analyzed using frequency counts.

The most frequent categories were visualized to understand the composition of the dataset.

Output:

```text
output/top10_category_counts.png
```

## Five EDA Observations

1. The dataset contains both numerical and categorical information, making it suitable for practicing different EDA techniques.

2. Numerical features show different distributions and ranges, which should be considered before applying machine learning algorithms.

3. The missing-value analysis using `isnull().sum()` helps identify whether any columns require additional cleaning or imputation.

4. The correlation analysis shows that some numerical variables have stronger relationships with student performance than others.

5. Categorical frequency analysis helps identify the most common groups in the dataset and can reveal imbalanced categories that may require attention during future modeling.

## EDA Narrative

The Indian Student Performance dataset provides useful information for understanding factors associated with student outcomes. The initial inspection using Pandas shows the structure of the dataset, including its rows, columns, data types, and descriptive statistics. Missing-value analysis is important because incomplete observations can affect both statistical analysis and future machine-learning models.

The distributions of numerical variables help identify the typical range of student-related measurements and reveal whether some variables are skewed or contain unusual observations. Such patterns should be considered during preprocessing, especially when algorithms are sensitive to feature scale or extreme values.

The correlation heatmap provides an overview of relationships between numerical variables. Features with stronger correlations may contain useful predictive information, while highly correlated input variables may introduce redundancy. Correlation, however, does not prove causation.

The categorical analysis highlights the most frequent categories and helps identify whether some groups are represented more heavily than others. Any strong imbalance should be considered before model training.

Overall, the EDA provides a foundation for data cleaning, feature engineering, and machine-learning modeling. Before building a predictive model, suspicious values, missing data, outliers, and categorical variables should be reviewed and handled appropriately.

## Output Evidence

The following files are generated by the analysis:

* `numeric_distributions.png` — distributions of numerical features
* `correlation_heatmap.png` — correlation matrix visualization
* `top10_category_counts.png` — top categorical values
* `eda_narrative.txt` — EDA narrative

## How to Run

### 1. Create and activate the virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the EDA script

```powershell
python eda.py
```

### 4. Check generated outputs

```powershell
Get-ChildItem output
```

## Requirements

The main Python libraries used in this project are:

```text
pandas
numpy
matplotlib
seaborn
```

## Git Workflow

### Branch

```text
feat/aiml-W1-Akshata
```

### Required Commits

```text
feat: eda — distributions, correlations, narrative
```

```text
docs: add EDA observations and documentation
```

A final documentation commit may also be added if required:

```text
docs: finalize W1D4 EDA completion
```

## W1D4 Completion

This project completes the W1D4 Exploratory Data Analysis task.

The analysis includes:

* Dataset inspection
* Missing-value analysis
* Descriptive statistics
* Numerical distributions
* Correlation analysis
* Categorical analysis
* Five EDA observations
* EDA narrative
* Output visualizations

## Conclusion

The EDA process provides an essential understanding of the dataset before machine-learning development. The analysis performed in this project helps identify data-quality issues, understand feature distributions, discover relationships between variables, and determine areas that may require preprocessing or further investigation.

This workflow demonstrates the practical use of Pandas and visualization libraries as part of the standard machine-learning data-preparation process.