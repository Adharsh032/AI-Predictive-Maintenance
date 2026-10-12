from flask import Blueprint, render_template

from database.models import get_dashboard_statistics


dashboard_bp = Blueprint(
    "dashboard",
    __name__
)


@dashboard_bp.route("/dashboard")
def dashboard():

    statistics = get_dashboard_statistics()

    return render_template(
        "dashboard.html",
        statistics=statistics
    )