import numpy as np
import pandas as pd


def array_fundamentals():
    """Create 1D, 2D, and 3D NumPy arrays and verify their shapes."""

    array_1d = np.array([10, 20, 30, 40, 50])

    array_2d = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    array_3d = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    print("=== Array Fundamentals ===")
    print("1D array:", array_1d)
    print("1D shape:", array_1d.shape)

    print("\n2D array:")
    print(array_2d)
    print("2D shape:", array_2d.shape)

    print("\n3D array:")
    print(array_3d)
    print("3D shape:", array_3d.shape)


def array_operations():
    """Demonstrate broadcasting, vectorised operations and matrix multiplication."""

    values = np.array([10, 20, 30, 40, 50])

    # Broadcasting
    broadcast_result = values + 5

    # Vectorised operations
    doubled = values * 2
    squared = values ** 2

    # Matrix multiplication
    matrix_a = np.array([
        [1, 2],
        [3, 4]
    ])

    matrix_b = np.array([
        [5, 6],
        [7, 8]
    ])

    matrix_result = matrix_a @ matrix_b

    print("\n=== Array Operations ===")
    print("Broadcasting (+5):", broadcast_result)
    print("Vectorised multiplication (*2):", doubled)
    print("Vectorised squaring:", squared)
    print("Matrix multiplication:")
    print(matrix_result)


def dataset_statistics(csv_path):
    """Calculate statistics using NumPy on numeric CSV data."""

    data = pd.read_csv(csv_path)
    numeric_data = data.select_dtypes(include=np.number)

    values = numeric_data.to_numpy()

    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0, ddof=1)
    correlation = np.corrcoef(values, rowvar=False)

    print("\n=== Dataset Statistics ===")

    print("\nColumns:")
    print(list(numeric_data.columns))

    print("\nMean:")
    print(mean_values)

    print("\nStandard deviation:")
    print(std_values)

    print("\nCorrelation:")
    print(correlation)


def main():
    """Run the NumPy fundamentals exercises."""

    array_fundamentals()
    array_operations()
    dataset_statistics("data/organizations.csv")

    print("\nNumPy fundamentals completed successfully.")


if __name__ == "__main__":
    main()