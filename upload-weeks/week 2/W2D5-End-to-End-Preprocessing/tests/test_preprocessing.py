```python
import os
import pandas as pd


DATA_PATH = "data/titanic.csv"
OUTPUT_PATH = "outputs/processed/titanic_ml_ready.csv"


def test_input_dataset_exists():
    """Check that the Titanic dataset exists."""
    assert os.path.exists(DATA_PATH)


def test_input_dataset_not_empty():
    """Check that the input dataset contains data."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0


def test_target_column_exists():
    """Check that the target column exists."""
    df = pd.read_csv(DATA_PATH)
    assert "Survived" in df.columns


def test_processed_file_exists():
    """Check that the processed dataset was created."""
    assert os.path.exists(OUTPUT_PATH)


def test_processed_dataset_has_no_missing_values():
    """Check that preprocessing removed all missing values."""
    df = pd.read_csv(OUTPUT_PATH)
    assert df.isnull().sum().sum() == 0


def test_processed_dataset_not_empty():
    """Check that the processed dataset contains rows."""
    df = pd.read_csv(OUTPUT_PATH)
    assert len(df) > 0


def test_target_values_are_valid():
    """Check that Survived contains only 0 and 1."""
    df = pd.read_csv(OUTPUT_PATH)
    assert set(df["Survived"].unique()).issubset({0, 1})