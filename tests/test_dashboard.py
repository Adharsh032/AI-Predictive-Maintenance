import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.models import get_dashboard_statistics


def main():

    statistics = get_dashboard_statistics()

    print("\n===== DASHBOARD STATISTICS =====")

    print(
        "Total Predictions:",
        statistics["total_predictions"]
    )

    print(
        "Total Failures:",
        statistics["total_failures"]
    )

    print(
        "Healthy:",
        statistics["healthy_machines"]
    )

    print(
        "Warning:",
        statistics["warning_machines"]
    )

    print(
        "Critical:",
        statistics["critical_machines"]
    )

    print(
        "\nPrediction Distribution:"
    )

    print(
        statistics["prediction_distribution"]
    )

    print(
        "\nHealth Distribution:"
    )

    print(
        statistics["health_distribution"]
    )

    print(
        "\nRecent Predictions:"
    )

    for prediction in statistics["recent_predictions"]:
        print(
            prediction
        )


if __name__ == "__main__":
    main()