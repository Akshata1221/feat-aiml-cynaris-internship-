# W1D5: Data Visualisation — Matplotlib & Seaborn

## Overview

This project demonstrates practical data visualization techniques using **Matplotlib** and **Seaborn** on the Indian Student Performance dataset.

The objective is to transform raw numerical and categorical data into meaningful visual insights that can support Exploratory Data Analysis and future machine-learning tasks.

## Objective

The project performs the following visualization tasks:

* Analyze numerical feature distributions.
* Identify relationships between numerical variables.
* Visualize correlations using a heatmap.
* Analyze the most frequent categorical values.
* Detect potential outliers using boxplots.
* Analyze the relationship between study hours and student scores.
* Save all visualizations as PNG files for evidence and documentation.

## Dataset

**File:**

```text
data/indian_student_performance.csv
```

The dataset contains student information including:

* Student ID
* Gender
* City
* Study Hours
* Math Score
* Science Score
* English Score
* Attendance Percentage
* Performance Level

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Project Structure

```text
W1D5 Data Visualisation/
│
├── data/
│   └── indian_student_performance.csv
│
├── output/
│   ├── *_distribution.png
│   ├── *_boxplot.png
│   ├── *_top10_categories.png
│   ├── correlation_heatmap.png
│   └── study_hours_score_trend.png
│
├── .gitignore
├── README.md
├── requirements.txt
└── visualisation.py
```

## Visualisations Created

### 1. Numeric Distribution Plots

Histograms with KDE curves were created for numerical variables.

These plots help identify:

* Data distribution
* Central tendency
* Spread
* Skewness
* Potential unusual values

Example outputs include:

```text
Math_Score_distribution.png
Science_Score_distribution.png
English_Score_distribution.png
Study_Hours_distribution.png
Attendance_Percent_distribution.png
```

### 2. Correlation Heatmap

A correlation matrix was calculated for numerical columns and visualized using a Seaborn heatmap.

Output:

```text
output/correlation_heatmap.png
```

The heatmap helps identify positive and negative relationships between numerical variables.

### 3. Top-10 Category Counts

Categorical variables were analyzed using frequency counts.

Bar plots were generated for categorical columns such as:

* Gender
* City
* Performance Level

Example outputs:

```text
Gender_top10_categories.png
City_top10_categories.png
Performance_Level_top10_categories.png
```

### 4. Boxplots

Boxplots were generated for numerical columns to identify:

* Median
* Spread
* Quartiles
* Potential outliers

Example outputs include:

```text
Math_Score_boxplot.png
Science_Score_boxplot.png
English_Score_boxplot.png
Study_Hours_boxplot.png
Attendance_Percent_boxplot.png
```

### 5. Study Hours Trend

A line plot was created to examine how average Math, Science, and English scores change with study hours.

Output:

```text
output/study_hours_score_trend.png
```

This visualization helps understand whether increased study time is associated with differences in average student scores.

## Key Observations

1. The score distributions provide an overview of student performance and show how scores are spread across the dataset.

2. Study hours show variation between students, making it useful for examining whether study time is associated with academic performance.

3. The correlation heatmap provides a quick way to identify numerical variables that have stronger or weaker relationships.

4. Boxplots make it easier to identify potential outliers that may require further investigation during data preprocessing.

5. The categorical plots show the frequency of different student groups and can reveal whether certain categories are more common than others.

## Design Decisions

The visualization code was designed to automatically detect numerical and categorical columns using Pandas.

Numerical columns are processed using:

```python
df.select_dtypes(include="number")
```

Categorical columns are identified using:

```python
df.select_dtypes(exclude="number")
```

This makes the script reusable if the dataset contains additional columns.

All plots are saved into the `output` directory rather than displayed only on screen. This provides persistent output evidence for the internship submission.

## How to Run

### 1. Activate the virtual environment

For Windows CMD:

```cmd
.venv\Scripts\activate
```

### 2. Install dependencies

```cmd
python -m pip install -r requirements.txt
```

### 3. Run the visualization script

```cmd
python visualisation.py
```

### 4. Check the generated files

```cmd
dir output
```

## Requirements

The project uses:

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

The project uses at least two descriptive commits:

```text
feat: add data visualisation analysis
```

and

```text
docs: add W1D5 documentation and evidence
```

## Deliverables

The completed W1D5 project contains:

* Working Python visualization code
* Indian student performance dataset
* Distribution plots
* Correlation heatmap
* Top-10 categorical plots
* Boxplots
* Study-hours trend visualization
* README documentation
* Output evidence
* Git branch and commits

## Self-Review Checklist

- [x] Loaded the Indian student performance dataset
- [x] Generated numeric distribution plots
- [x] Generated correlation heatmap
- [x] Generated categorical analysis plots
- [x] Generated boxplots
- [x] Generated study-hours score trend
- [x] Saved visualization outputs
- [x] Used clean and commented Python code
- [x] Tested the script successfully
- [x] Completed Git workflow

## Conclusion

The project demonstrates how Matplotlib and Seaborn can transform raw student-performance data into useful visual insights.

Distribution plots help understand individual variables, correlation heatmaps reveal relationships, categorical plots summarize group frequencies, boxplots highlight potential outliers, and the study-hours trend provides a visual comparison between study time and average academic scores.

These visualization techniques form an important part of Exploratory Data Analysis and provide a foundation for data cleaning, feature engineering, and machine-learning model development.
