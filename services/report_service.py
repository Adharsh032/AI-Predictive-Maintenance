from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.units import inch


class ReportService:

    def __init__(self):

        project_root = Path(__file__).resolve().parent.parent

        self.reports_dir = project_root / "reports"

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def generate_report(self, prediction):

        prediction_id = prediction["id"]

        filename = (
            f"maintenance_report_{prediction_id}.pdf"
        )

        file_path = self.reports_dir / filename

        document = SimpleDocTemplate(
            str(file_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            spaceAfter=20,
        )

        heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            spaceBefore=12,
            spaceAfter=8,
        )

        body_style = styles["BodyText"]

        story = []

        # Title
        story.append(
            Paragraph(
                "AI Predictive Maintenance Report",
                title_style
            )
        )

        story.append(
            Paragraph(
                f"Prediction ID: {prediction_id}",
                body_style
            )
        )

        story.append(Spacer(1, 15))

        # Prediction summary
        story.append(
            Paragraph(
                "Prediction Summary",
                heading_style
            )
        )

        prediction_text = (
            "Machine Failure"
            if prediction["prediction"] == 1
            else "No Machine Failure"
        )

        summary_data = [
            ["Prediction", prediction_text],
            [
                "Failure Probability",
                f"{prediction['failure_probability'] * 100:.2f}%"
            ],
            [
                "Machine Health",
                prediction.get(
                    "health_status",
                    "Not Available"
                )
            ],
            [
                "AI Method",
                prediction.get(
                    "ai_method",
                    "Not Available"
                )
            ],
            [
                "Date & Time",
                str(
                    prediction.get(
                        "created_at",
                        "Not Available"
                    )
                )
            ],
        ]

        summary_table = Table(
            summary_data,
            colWidths=[2.2 * inch, 4.2 * inch]
        )

        summary_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(summary_table)

        # Sensor values
        story.append(
            Paragraph(
                "Machine Sensor Values",
                heading_style
            )
        )

        sensor_data = [
            [
                "Parameter",
                "Value"
            ],
            [
                "Product Type",
                str(prediction["product_type"])
            ],
            [
                "Air Temperature",
                f"{prediction['air_temperature']} K"
            ],
            [
                "Process Temperature",
                f"{prediction['process_temperature']} K"
            ],
            [
                "Rotational Speed",
                f"{prediction['rotational_speed']} rpm"
            ],
            [
                "Torque",
                f"{prediction['torque']} Nm"
            ],
            [
                "Tool Wear",
                f"{prediction['tool_wear']} min"
            ],
        ]

        sensor_table = Table(
            sensor_data,
            colWidths=[2.2 * inch, 4.2 * inch]
        )

        sensor_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.darkgrey
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(sensor_table)

        # AI analysis
        story.append(
            Paragraph(
                "AI Maintenance Analysis",
                heading_style
            )
        )

        ai_analysis = prediction.get(
            "ai_analysis"
        )

        if ai_analysis:

            # Convert stored analysis into safe PDF paragraphs
            analysis_text = str(ai_analysis)

            for line in analysis_text.splitlines():

                line = line.strip()

                if line:

                    story.append(
                        Paragraph(
                            line.replace(
                                "&",
                                "&amp;"
                            ),
                            body_style
                        )
                    )

                    story.append(
                        Spacer(1, 5)
                    )

        else:

            story.append(
                Paragraph(
                    "AI maintenance analysis is not available "
                    "for this prediction.",
                    body_style
                )
            )

        # Safety note
        story.append(
            Spacer(1, 10)
        )

        story.append(
            Paragraph(
                "<b>Safety Note:</b> This system is a "
                "decision-support prototype. Maintenance "
                "decisions should be validated by qualified "
                "maintenance personnel.",
                body_style
            )
        )

        document.build(story)

        return file_path
        