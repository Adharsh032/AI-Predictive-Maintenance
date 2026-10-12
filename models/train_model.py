from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "lightgbm_model.pkl"
)

PREPROCESSING_PATH = (
    MODEL_DIR
    / "preprocessing.pkl"
)


# ---------------------------------------------------------
# Features and target
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


# ---------------------------------------------------------
# Load training and testing data
# ---------------------------------------------------------

def load_data():

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA_PATH}"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Testing data not found: {TEST_DATA_PATH}"
        )

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    return train_df, test_df


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

def prepare_data(train_df, test_df):

    X_train = train_df[
        FEATURE_COLUMNS
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].copy()

    X_test = test_df[
        FEATURE_COLUMNS
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ---------------------------------------------------------
# Create preprocessing pipeline
# ---------------------------------------------------------

def create_preprocessor():

    categorical_features = [
        "Type"
    ]

    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Calculate class weight
# ---------------------------------------------------------

def calculate_scale_pos_weight(y_train):

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    return scale_pos_weight


# ---------------------------------------------------------
# Create LightGBM model
# ---------------------------------------------------------

def create_model(scale_pos_weight):

    model = lgb.LGBMClassifier(

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

    return model


# ---------------------------------------------------------
# Create complete ML pipeline
# ---------------------------------------------------------

def create_pipeline(
    preprocessor,
    model,
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(
    pipeline,
    X_test,
    y_test,
):

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------
# Save model and preprocessing
# ---------------------------------------------------------

def save_artifacts(
    pipeline,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(
        f"\nComplete pipeline saved to:"
    )

    print(
        MODEL_PATH
    )

    # Save preprocessing separately
    preprocessing = pipeline.named_steps[
        "preprocessor"
    ]

    joblib.dump(
        preprocessing,
        PREPROCESSING_PATH
    )

    print(
        f"Preprocessing pipeline saved to:"
    )

    print(
        PREPROCESSING_PATH
    )


# ---------------------------------------------------------
# Main training process
# ---------------------------------------------------------

def train_model():

    print("=" * 60)
    print("LIGHTGBM MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n1. Loading training and testing data...")

    train_df, test_df = load_data()

    print(
        f"Training data shape: {train_df.shape}"
    )

    print(
        f"Testing data shape: {test_df.shape}"
    )

    # Prepare data
    print("\n2. Preparing features and target...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(
        train_df,
        test_df
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # Class distribution
    print("\n3. Checking class distribution...")

    print(
        "Training class distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    # Calculate class weight
    print("\n4. Calculating class imbalance weight...")

    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # Preprocessor
    print("\n5. Creating preprocessing pipeline...")

    preprocessor = create_preprocessor()

    # Model
    print("\n6. Creating LightGBM model...")

    model = create_model(
        scale_pos_weight
    )

    # Complete pipeline
    pipeline = create_pipeline(
        preprocessor,
        model
    )

    # Train
    print("\n7. Training LightGBM...")

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # Evaluate
    print("\n8. Evaluating model...")

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    # Save
    print("\n9. Saving model artifacts...")

    save_artifacts(
        pipeline
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    return metrics


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    train_model()