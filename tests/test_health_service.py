import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.health_service import HealthService


def main():

    service = HealthService()

    test_values = [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90
    ]

    print("\nMachine Health Test:\n")

    for probability in test_values:

        status = service.get_health_status(
            probability
        )

        print(
            f"Failure Probability: "
            f"{probability * 100:.1f}% "
            f"-> Health: {status}"
        )


if __name__ == "__main__":
    main()