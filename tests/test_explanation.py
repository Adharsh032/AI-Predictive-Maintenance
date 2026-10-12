import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.explanation_service import ExplanationService


def main():

    service = ExplanationService()

    input_data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0
    }

    result = service.explain_prediction(input_data)

    print("\nExplanation Method:")
    print(result["method"])

    print("\nImportant Features:")

    for feature in result["features"]:

        print(
            f"{feature['feature']} | "
            f"SHAP: {feature['shap_value']:.6f} | "
            f"{feature['direction']}"
        )


if __name__ == "__main__":
    main()