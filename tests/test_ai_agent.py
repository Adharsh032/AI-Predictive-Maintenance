import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.ai_agent import AIAgent


def main():

    agent = AIAgent()

    input_data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 312.0,
        "Rotational speed [rpm]": 2100.0,
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 160.0
    }

    prediction = 1

    failure_probability = 0.85

    explanation = {
        "method": "SHAP",
        "features": [
            {
                "feature": "Torque [Nm]",
                "shap_value": 1.2,
                "direction": "Increases failure risk"
            },
            {
                "feature": "Tool wear [min]",
                "shap_value": 0.8,
                "direction": "Increases failure risk"
            }
        ]
    }

    maintenance_knowledge = [
        {
            "condition": "High torque",
            "possible_causes": [
                "Excessive mechanical load",
                "Tool wear"
            ],
            "recommended_actions": [
                "Inspect the tool",
                "Check for excessive load"
            ],
            "priority": "High"
        }
    ]

    result = agent.analyze(
        input_data,
        prediction,
        failure_probability,
        explanation,
        maintenance_knowledge
    )

    print("\nAI Agent Method:")
    print(result["method"])

    print("\nAI Agent Status:")
    print(result["status"])

    print("\nAI Agent Analysis:")
    print(result["analysis"])


if __name__ == "__main__":
    main()