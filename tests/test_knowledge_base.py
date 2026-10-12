import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.knowledge_base import KnowledgeBase


def main():

    kb = KnowledgeBase()

    input_data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 312.0,
        "Rotational speed [rpm]": 2100.0,
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 160.0
    }

    prediction = 1

    results = kb.retrieve(
        input_data,
        prediction
    )

    print("\nRelevant Maintenance Knowledge:\n")

    for item in results:

        print("Condition:", item["condition"])

        print("Possible Causes:")
        for cause in item["possible_causes"]:
            print("-", cause)

        print("Recommended Actions:")
        for action in item["recommended_actions"]:
            print("-", action)

        print("Priority:", item["priority"])

        print("-" * 50)


if __name__ == "__main__":
    main()