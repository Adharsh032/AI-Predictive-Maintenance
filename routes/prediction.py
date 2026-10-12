import time
from flask import Blueprint, render_template, request

from services.prediction_service import PredictionService
from services.explanation_service import ExplanationService
from services.knowledge_base import KnowledgeBase
from services.ai_agent import AIAgent
from services.health_service import HealthService

from database.models import (
    save_prediction,
    get_all_predictions,
    get_prediction_by_id,
)


prediction_bp = Blueprint(
    "prediction",
    __name__,
    url_prefix="/predict"
)


prediction_service = PredictionService()
explanation_service = ExplanationService()
knowledge_base = KnowledgeBase()
ai_agent = AIAgent()
health_service = HealthService()

@prediction_bp.route("/", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@prediction_bp.route("/", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        input_data = {
            "Type": request.form.get("Type"),

            "Air temperature [K]": float(
                request.form.get("Air temperature [K]")
            ),

            "Process temperature [K]": float(
                request.form.get("Process temperature [K]")
            ),

            "Rotational speed [rpm]": float(
                request.form.get("Rotational speed [rpm]")
            ),

            "Torque [Nm]": float(
                request.form.get("Torque [Nm]")
            ),

            "Tool wear [min]": float(
                request.form.get("Tool wear [min]")
            ),
        }
        # Validate reasonable sensor ranges

        if not 290 <= input_data["Air temperature [K]"] <= 305:
            raise ValueError(
                "Air Temperature must be between 290 K and 305 K."
            )

        if not 300 <= input_data["Process temperature [K]"] <= 315:
            raise ValueError(
                "Process Temperature must be between 300 K and 315 K."
            )

        if not 1000 <= input_data["Rotational speed [rpm]"] <= 3000:
            raise ValueError(
                "Rotational Speed must be between 1000 rpm and 3000 rpm."
            )

        if not 10 <= input_data["Torque [Nm]"] <= 80:
            raise ValueError(
                "Torque must be between 10 Nm and 80 Nm."
            )

        if not 0 <= input_data["Tool wear [min]"] <= 300:
            raise ValueError(
                "Tool Wear must be between 0 and 300 minutes."
            )
        # LightGBM prediction
        result = prediction_service.predict(input_data)
        print(
            f"LightGBM time: "
            f"{time.time() - start_time:.2f} seconds"
        )

        # SHAP explanation
        explanation = explanation_service.explain_prediction(
            input_data
        )
        print(
            f"SHAP time: "
            f"{time.time() - start_time:.2f} seconds"
        )
        # Retrieve relevant maintenance knowledge
        maintenance_knowledge = knowledge_base.retrieve(
            input_data,
            result["prediction"]
        )
        print(
            f"Knowledge Base time: "
            f"{time.time() - start_time:.2f} seconds"
        )
        # AI Agent analysis
        ai_analysis = ai_agent.analyze(
            input_data=input_data,
            prediction=result["prediction"],
            failure_probability=result["failure_probability"],
            explanation=explanation,
            maintenance_knowledge=maintenance_knowledge
        )
        # Determine machine health status
        health_status = health_service.get_health_status(
            result["failure_probability"]
        )
        print(
            f"AI Agent time: "
            f"{time.time() - start_time:.2f} seconds"
        )

        # Save prediction to database
        prediction_id = save_prediction(
            product_type=input_data["Type"],
            air_temperature=input_data["Air temperature [K]"],
            process_temperature=input_data["Process temperature [K]"],
            rotational_speed=input_data["Rotational speed [rpm]"],
            torque=input_data["Torque [Nm]"],
            tool_wear=input_data["Tool wear [min]"],
            prediction=result["prediction"],
            failure_probability=result["failure_probability"],
            health_status=health_status,
            ai_method=ai_analysis["method"],
            ai_analysis=(
                ai_analysis["analysis"]
                if isinstance(ai_analysis["analysis"], str)
                else str(ai_analysis["analysis"])
            ),
        )

        # Display result + SHAP explanation
        return render_template(
            "result.html",
            input_data=input_data,
            result=result,
            prediction_id=prediction_id,
            explanation=explanation,
            maintenance_knowledge=maintenance_knowledge,
            ai_analysis=ai_analysis,
            health_status=health_status
        )

    except (ValueError, TypeError) as error:

        return render_template(
            "predict.html",
            error=str(error)
        )

    except Exception as error:

        return render_template(
            "predict.html",
            error=f"Prediction error: {error}"
        )


@prediction_bp.route("/history/<int:prediction_id>", methods=["GET"])
def prediction_detail(prediction_id):

    prediction = get_prediction_by_id(prediction_id)

    if prediction is None:
        return "Prediction not found", 404

    return render_template(
        "history_detail.html",
        prediction=prediction
    )