import json
import requests

from config import OLLAMA_URL, OLLAMA_MODEL


class AIAgent:

    def __init__(self):
        self.ollama_url = OLLAMA_URL
        self.model = OLLAMA_MODEL

    def analyze(
        self,
        input_data,
        prediction,
        failure_probability,
        explanation,
        maintenance_knowledge
    ):
        prompt = self._build_prompt(
            input_data,
            prediction,
            failure_probability,
            explanation,
            maintenance_knowledge
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 250
                    }
                },
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            llm_response = data.get("response", "").strip()

            if not llm_response:
                raise ValueError("Empty response from Ollama")

            return {
                "method": "Llama 3",
                "status": "success",
                "analysis": llm_response
            }

        except Exception as error:

            print("\nOLLAMA ERROR:")
            print(type(error).__name__)
            print(str(error))

            return self._fallback_response(
                prediction,
                failure_probability,
                explanation,
                maintenance_knowledge
            )

    def _build_prompt(
        self,
        input_data,
        prediction,
        failure_probability,
        explanation,
        maintenance_knowledge
    ):

        prediction_text = (
            "Machine Failure"
            if prediction == 1
            else "No Machine Failure"
        )

        prompt = f"""
You are a predictive maintenance AI assistant.

Your job is ONLY to interpret the existing LightGBM prediction.

RULES:
- Do not make a new prediction.
- Do not change the LightGBM prediction.
- Do not invent sensor values or probabilities.
- Use the supplied SHAP explanation.
- Use the supplied maintenance knowledge.
- Give general maintenance recommendations.
- Recommend validation by qualified maintenance personnel.
- Do not use unsupported claims such as catastrophic failure.

LIGHTGBM PREDICTION:
{prediction_text}

FAILURE PROBABILITY:
{failure_probability:.4f}

SENSOR VALUES:
{json.dumps(input_data)}

SHAP:
{json.dumps(explanation)}

MAINTENANCE KNOWLEDGE:
{json.dumps(maintenance_knowledge)}

Provide a concise analysis using exactly these sections:

Prediction Summary:
Explanation:
Possible Causes:
Important Contributing Factors:
Maintenance Recommendations:
Priority:
Safety Note:

Keep the complete response concise.
"""

        return prompt

    def _fallback_response(
        self,
        prediction,
        failure_probability,
        explanation,
        maintenance_knowledge
    ):

        prediction_text = (
            "Machine Failure"
            if prediction == 1
            else "No Machine Failure"
        )

        causes = []
        recommendations = []
        priorities = []

        for item in maintenance_knowledge:

            causes.extend(
                item.get("possible_causes", [])
            )

            recommendations.extend(
                item.get("recommended_actions", [])
            )

            priorities.append(
                item.get("priority", "Medium")
            )

        priority = "Medium"

        if "Critical" in priorities:
            priority = "Critical"
        elif "High" in priorities:
            priority = "High"

        important_factors = []

        for feature in explanation.get("features", []):

            important_factors.append(
                feature.get("feature", "Unknown feature")
            )

        return {
            "method": "Knowledge Base Fallback",
            "status": "fallback",
            "analysis": {
                "prediction_summary": prediction_text,
                "explanation": (
                    "The result is based on the LightGBM "
                    "prediction and available explanation."
                ),
                "possible_causes": list(dict.fromkeys(causes)),
                "important_contributing_factors": important_factors,
                "maintenance_recommendations": (
                    list(dict.fromkeys(recommendations))
                ),
                "priority": priority,
                "safety_note": (
                    "This is a decision-support system. "
                    "Validate the condition with qualified "
                    "maintenance personnel before taking action."
                )
            }
        }