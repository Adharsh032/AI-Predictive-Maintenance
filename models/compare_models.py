
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# Project paths and dataset configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "model_comparison_results.csv"
)

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_COLUMNS = ["Type"]
TARGET_COLUMN = "Machine failure"


# ---------------------------------------------------------
# Load the existing train/test split
# ---------------------------------------------------------

def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    for name, df in [("training", train_df), ("test", test_df)]:
        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name} data: {sorted(missing)}"
            )

        if df[required_columns].isnull().any().any():
            raise ValueError(
                f"Missing values found in {name} data. "
                "Investigate preprocessing before comparison."
            )

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].astype(int)

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain both classes 0 and 1.")

    if set(y_test.unique()) != {0, 1}:
        raise ValueError("Test target must contain both classes 0 and 1.")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def create_preprocessor(scale_numeric=False):
    numeric_transformer = (
        StandardScaler() if scale_numeric else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numerical",
                numeric_transformer,
                NUMERIC_COLUMNS,
            ),
        ]
    )


def create_pipeline(model, scale_numeric=False):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(scale_numeric=scale_numeric),
            ),
            ("model", model),
        ]
    )


# ---------------------------------------------------------
# Define models
# ---------------------------------------------------------

def create_models(y_train):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    return {
        "Logistic Regression": create_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
            scale_numeric=True,
        ),

        "Random Forest": create_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
        ),

        "SVM": create_pipeline(
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
            scale_numeric=True,
        ),

        "LightGBM": create_pipeline(
            lgb.LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
    }


# ---------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------

def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    print(f"\nTraining and evaluating: {name}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_Failure": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall_Failure": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1_Failure": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, y_probability),
        "PR_Average_Precision": average_precision_score(
            y_test, y_probability
        ),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print(f"  Accuracy:          {metrics['Accuracy']:.4f}")
    print(f"  Failure precision: {metrics['Precision_Failure']:.4f}")
    print(f"  Failure recall:    {metrics['Recall_Failure']:.4f}")
    print(f"  Failure F1-score:  {metrics['F1_Failure']:.4f}")
    print(f"  ROC-AUC:           {metrics['ROC_AUC']:.4f}")
    print(
        "  Confusion matrix [[TN, FP], [FN, TP]]: "
        f"[[{tn}, {fp}], [{fn}, {tp}]]"
    )

    return metrics


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("PREDICTIVE MAINTENANCE: MODEL COMPARISON")
    print("=" * 65)

    X_train, X_test, y_train, y_test = load_data()

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Training failures: {(y_train == 1).sum()}")
    print(f"Test failures:     {(y_test == 1).sum()}")

    models = create_models(y_train)
    results = []

    for name, pipeline in models.items():
        results.append(
            evaluate_model(
                name,
                pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
            )
        )

    results_df = pd.DataFrame(results)

    # Sort for convenient inspection; this is not an automatic
    # declaration of the best model.
    results_df = results_df.sort_values(
        by="F1_Failure", ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    display_columns = [
        "Model",
        "Accuracy",
        "Precision_Failure",
        "Recall_Failure",
        "F1_Failure",
        "ROC_AUC",
        "PR_Average_Precision",
        "Balanced_Accuracy",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nFull results saved to:\n{OUTPUT_PATH}")
    print("\nComparison completed. Review metrics before drawing conclusions.")


if __name__ == "__main__":
    main()
