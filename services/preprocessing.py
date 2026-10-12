from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"


COLUMNS_TO_DROP = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset():
    """
    Load the original AI4I dataset from the raw data folder.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    return df


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

def validate_columns(df):
    """
    Check whether all required dataset columns exist.
    """

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + COLUMNS_TO_DROP
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )


# ---------------------------------------------------------
# Clean dataset
# ---------------------------------------------------------

def clean_dataset(df):
    """
    Clean the dataset and remove unnecessary /
    target-leaking columns.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Keep only the required model features and target
    columns_to_keep = FEATURE_COLUMNS + [TARGET_COLUMN]

    df = df[columns_to_keep]

    # Handle missing numeric values
    numeric_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(
            df[column].median()
        )

    # Handle missing categorical values
    df["Type"] = df["Type"].fillna(
        df["Type"].mode()[0]
    )

    # Handle missing target values
    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    # Ensure target is integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# ---------------------------------------------------------
# Validate values
# ---------------------------------------------------------

def validate_values(df):
    """
    Validate important values in the dataset.
    """

    valid_product_types = {"L", "M", "H"}

    invalid_types = set(df["Type"].unique()) - valid_product_types

    if invalid_types:
        raise ValueError(
            f"Invalid product types found: {invalid_types}"
        )

    if (df["Machine failure"] < 0).any():
        raise ValueError(
            "Machine failure contains invalid negative values."
        )

    if not df["Machine failure"].isin([0, 1]).all():
        raise ValueError(
            "Machine failure must contain only 0 and 1."
        )

    numeric_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    for column in numeric_columns:

        if df[column].isna().any():
            raise ValueError(
                f"{column} still contains missing values."
            )

        if (df[column] < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )


# ---------------------------------------------------------
# Encode product type
# ---------------------------------------------------------

def encode_product_type(df):
    """
    Convert Product Type into numerical values.

    L = 0
    M = 1
    H = 2
    """

    df = df.copy()

    type_mapping = {
        "L": 0,
        "M": 1,
        "H": 2,
    }

    df["Type"] = df["Type"].map(type_mapping)

    if df["Type"].isna().any():
        raise ValueError(
            "Product Type encoding produced missing values."
        )

    return df


# ---------------------------------------------------------
# Create train/test datasets
# ---------------------------------------------------------

def create_train_test_data(df):
    """
    Split the processed dataset into training and testing
    sets while preserving the failure class distribution.
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Encode product type
    X = encode_product_type(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Save processed data
# ---------------------------------------------------------

def save_processed_data(
    cleaned_df,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Save cleaned and split datasets.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned_path = (
        PROCESSED_DATA_DIR
        / "ai4i2020_cleaned.csv"
    )

    train_path = (
        PROCESSED_DATA_DIR
        / "train.csv"
    )

    test_path = (
        PROCESSED_DATA_DIR
        / "test.csv"
    )

    train_combined = X_train.copy()
    train_combined[TARGET_COLUMN] = y_train.values

    test_combined = X_test.copy()
    test_combined[TARGET_COLUMN] = y_test.values

    cleaned_df.to_csv(
        cleaned_path,
        index=False
    )

    train_combined.to_csv(
        train_path,
        index=False
    )

    test_combined.to_csv(
        test_path,
        index=False
    )

    return (
        cleaned_path,
        train_path,
        test_path,
    )


# ---------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------

def run_preprocessing():
    """
    Execute the complete dataset preprocessing pipeline.
    """

    print("=" * 60)
    print("AI4I DATASET PREPROCESSING")
    print("=" * 60)

    print("\n1. Loading dataset...")

    df = load_dataset()

    print(
        f"Original dataset shape: {df.shape}"
    )

    print("\n2. Validating columns...")

    validate_columns(df)

    print("Required columns found.")

    print("\n3. Cleaning dataset...")

    cleaned_df = clean_dataset(df)

    print(
        f"Cleaned dataset shape: {cleaned_df.shape}"
    )

    print("\n4. Validating values...")

    validate_values(cleaned_df)

    print("Value validation successful.")

    print("\n5. Creating train/test split...")

    X_train, X_test, y_train, y_test = (
        create_train_test_data(cleaned_df)
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print("\n6. Saving processed datasets...")

    paths = save_processed_data(
        cleaned_df,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nFiles created:")

    for path in paths:
        print(f" - {path}")

    print("\n7. Class distribution:")

    print(
        cleaned_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\n8. Processing completed successfully.")

    print("=" * 60)


# ---------------------------------------------------------
# Run script
# ---------------------------------------------------------

if __name__ == "__main__":
    run_preprocessing()