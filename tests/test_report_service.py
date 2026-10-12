import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.models import get_all_predictions
from services.report_service import ReportService


def main():

    predictions = get_all_predictions()

    if not predictions:

        print("No predictions available.")
        return

    prediction = predictions[0]

    service = ReportService()

    report_path = service.generate_report(
        prediction
    )

    print("\nPDF report generated successfully.")
    print("Report path:")
    print(report_path)


if __name__ == "__main__":
    main()