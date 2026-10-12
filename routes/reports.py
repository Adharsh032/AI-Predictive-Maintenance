from flask import Blueprint, redirect, url_for, send_file, abort
from pathlib import Path

from database.models import get_prediction_by_id
from services.report_service import ReportService


reports_bp = Blueprint("reports", __name__)

report_service = ReportService()


@reports_bp.route(
    "/report/<int:prediction_id>",
    methods=["POST"]
)
def generate_report(prediction_id):

    prediction = get_prediction_by_id(prediction_id)

    if prediction is None:
        abort(404)

    # Generate the PDF
    report_service.generate_report(prediction)

    # Redirect to the download route
    return redirect(
        url_for(
            "reports.download_report",
            prediction_id=prediction_id
        )
    )


@reports_bp.route(
    "/report/<int:prediction_id>/download",
    methods=["GET"]
)
def download_report(prediction_id):

    prediction = get_prediction_by_id(prediction_id)

    if prediction is None:
        abort(404)

    # Check whether the PDF already exists
    project_root = Path(__file__).resolve().parent.parent

    report_path = (
        project_root
        / "reports"
        / f"maintenance_report_{prediction_id}.pdf"
    )

    if not report_path.exists():
        return (
            "Report has not been generated yet. "
            "Please click Generate PDF Report first.",
            404
        )

    return send_file(
        report_path,
        as_attachment=True,
        download_name=report_path.name,
        mimetype="application/pdf"
    )