
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

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

NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_dataset():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def validate_columns(df):
    required = FEATURE_COLUMNS + [TARGET_COLUMN] + COLUMNS_TO_DROP
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataset(df):
    df = df.copy().drop_duplicates()
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Keep missing feature values for later train-fitted imputation.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Type"] = df["Type"].where(
        df["Type"].isin(["L", "M", "H"]), pd.NA
    )

    # A target label cannot be safely imputed.
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Machine failure must contain only 0 and 1.")

    return df


def validate_values(df):
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Machine failure must contain only 0 and 1.")

    if not df["Type"].dropna().isin(["L", "M", "H"]).all():
        raise ValueError("Type must contain only L, M, or H.")

    for column in NUMERIC_COLUMNS:
        values = df[column].dropna()
        if not values.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all():
            raise ValueError(f"{column} contains non-finite values.")
        if (values < 0).any():
            raise ValueError(f"{column} contains negative values.")


def create_train_test_data(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Fit imputation statistics on training data only.
    train_medians = X_train[NUMERIC_COLUMNS].median()
    train_modes = X_train["Type"].mode()

    if train_modes.empty:
        raise ValueError("Cannot impute Type: training set has no valid values.")

    type_mode = train_modes.iloc[0]

    X_train.loc[:, NUMERIC_COLUMNS] = X_train[NUMERIC_COLUMNS].fillna(
        train_medians
    )
    X_test.loc[:, NUMERIC_COLUMNS] = X_test[NUMERIC_COLUMNS].fillna(
        train_medians
    )

    X_train.loc[:, "Type"] = X_train["Type"].fillna(type_mode)
    X_test.loc[:, "Type"] = X_test["Type"].fillna(type_mode)

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing feature values remain after imputation.")

    return X_train, X_test, y_train, y_test


def save_processed_data(cleaned_df, X_train, X_test, y_train, y_test):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = PROCESSED_DATA_DIR / "ai4i2020_cleaned.csv"
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_combined = X_train.copy()
    train_combined[TARGET_COLUMN] = y_train

    test_combined = X_test.copy()
    test_combined[TARGET_COLUMN] = y_test

    cleaned_df.to_csv(cleaned_path, index=False)
    train_combined.to_csv(train_path, index=False)
    test_combined.to_csv(test_path, index=False)

    return cleaned_path, train_path, test_path


def run_preprocessing():
    print("=" * 60)
    print("AI4I DATASET PREPROCESSING")
    print("=" * 60)

    df = load_dataset()
    print(f"Original dataset shape: {df.shape}")

    validate_columns(df)

    cleaned_df = clean_dataset(df)
    validate_values(cleaned_df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")

    X_train, X_test, y_train, y_test = create_train_test_data(cleaned_df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training failures: {(y_train == 1).sum()}")
    print(f"Testing failures: {(y_test == 1).sum()}")

    paths = save_processed_data(
        cleaned_df, X_train, X_test, y_train, y_test
    )

    print("\nFiles created:")
    for path in paths:
        print(f" - {path}")

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    run_preprocessing()
