from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_model.pkl"


FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


class PredictionService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained LightGBM pipeline."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def validate_input(self, input_data):
        """Validate machine sensor input."""

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in input_data
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        valid_types = ["L", "M", "H"]

        if input_data["Type"] not in valid_types:
            raise ValueError(
                "Type must be one of: L, M, H"
            )

        numeric_features = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        for feature in numeric_features:
            try:
                float(input_data[feature])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{feature} must be numeric"
                )

        return True

    def prepare_input(self, input_data):
        """Convert input dictionary into a DataFrame."""

        self.validate_input(input_data)

        data = {
            feature: [input_data[feature]]
            for feature in FEATURE_COLUMNS
        }

        return pd.DataFrame(data)

    def predict(self, input_data):
        """Generate machine failure prediction."""

        X = self.prepare_input(input_data)

        prediction = int(self.model.predict(X)[0])

        probability = float(
            self.model.predict_proba(X)[0][1]
        )

        return {
            "prediction": prediction,
            "failure_probability": probability,
            "failure_percentage": probability * 100,
        }


def create_prediction_service():
    """Create and return a PredictionService instance."""

    return PredictionService()


if __name__ == "__main__":
    service = create_prediction_service()

    sample_input = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }

    result = service.predict(sample_input)

    print("=" * 60)
    print("PREDICTION SERVICE TEST")
    print("=" * 60)

    print(f"\nPrediction: {result['prediction']}")
    print(
        f"Failure Probability: "
        f"{result['failure_probability']:.4f}"
    )
    print(
        f"Failure Percentage: "
        f"{result['failure_percentage']:.2f}%"
    )