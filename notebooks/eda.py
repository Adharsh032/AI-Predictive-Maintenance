from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i2020_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "eda"
)


# ---------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    """Load the cleaned AI4I dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

def create_output_directory():
    """Create the EDA output directory."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

def generate_summary(df):
    """Print basic information about the dataset."""

    print("=" * 60)
    print("AI4I 2020 DATASET - EDA")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe())


# ---------------------------------------------------------
# Class distribution
# ---------------------------------------------------------

def plot_class_distribution(df):
    """Create machine failure class distribution plot."""

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=df,
        x=TARGET_COLUMN
    )

    ax.set_title(
        "Machine Failure Class Distribution"
    )

    ax.set_xlabel(
        "Machine Failure"
    )

    ax.set_ylabel(
        "Number of Samples"
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------
# Sensor distributions
# ---------------------------------------------------------

def plot_sensor_distributions(df):
    """Create distributions for numeric sensor features."""

    for feature in NUMERIC_FEATURES:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True
        )

        plt.title(
            f"Distribution of {feature}"
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        safe_name = (
            feature
            .lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        output_path = (
            OUTPUT_DIR
            / f"{safe_name}_distribution.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


# ---------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------

def plot_correlation_matrix(df):
    """Create a correlation matrix for numeric variables."""

    correlation_columns = (
        NUMERIC_FEATURES
        + [TARGET_COLUMN]
    )

    correlation = df[
        correlation_columns
    ].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Correlation Matrix"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "correlation_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------
# Failure vs sensor comparisons
# ---------------------------------------------------------

def plot_failure_vs_sensors(df):
    """Compare sensor values for failure and non-failure cases."""

    for feature in NUMERIC_FEATURES:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feature
        )

        plt.title(
            f"{feature} vs Machine Failure"
        )

        plt.xlabel(
            "Machine Failure"
        )

        plt.ylabel(feature)

        plt.tight_layout()

        safe_name = (
            feature
            .lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        output_path = (
            OUTPUT_DIR
            / f"failure_vs_{safe_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


# ---------------------------------------------------------
# Product type vs failure
# ---------------------------------------------------------

def plot_product_type_failure(df):
    """Compare machine failures across product types."""

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=df,
        x="Type",
        hue=TARGET_COLUMN
    )

    ax.set_title(
        "Machine Failure by Product Type"
    )

    ax.set_xlabel(
        "Product Type"
    )

    ax.set_ylabel(
        "Number of Samples"
    )

    plt.legend(
        title="Machine Failure"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "product_type_failure.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------
# Failure statistics
# ---------------------------------------------------------

def print_failure_statistics(df):
    """Print useful statistics for failure analysis."""

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    print("\nMachine failure counts:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nMachine failure percentages:")

    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        percentages.round(2)
    )

    print("\nNumeric feature means by failure class:")

    means = (
        df.groupby(TARGET_COLUMN)[
            NUMERIC_FEATURES
        ]
        .mean()
    )

    print(
        means.round(3)
    )


# ---------------------------------------------------------
# Main EDA pipeline
# ---------------------------------------------------------

def run_eda():

    create_output_directory()

    print("\nLoading dataset...")

    df = load_data()

    print("Dataset loaded successfully.")

    generate_summary(df)

    print("\nGenerating class distribution...")

    plot_class_distribution(df)

    print("\nGenerating sensor distributions...")

    plot_sensor_distributions(df)

    print("\nGenerating correlation matrix...")

    plot_correlation_matrix(df)

    print("\nGenerating failure vs sensor comparisons...")

    plot_failure_vs_sensors(df)

    print("\nGenerating product type comparison...")

    plot_product_type_failure(df)

    print_failure_statistics(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ---------------------------------------------------------
# Run EDA
# ---------------------------------------------------------

if __name__ == "__main__":
    run_eda()