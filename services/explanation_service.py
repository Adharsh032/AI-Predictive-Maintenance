import joblib
import pandas as pd
import numpy as np
import shap

from config import MODEL_PATH


class ExplanationService:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def explain_prediction(self, input_data):
        """
        Generate SHAP explanations for a single prediction.
        """

        try:
            # Convert input dictionary to DataFrame
            input_df = pd.DataFrame([input_data])

            # Get preprocessing pipeline
            preprocessor = self.model.named_steps["preprocessor"]

            # Get LightGBM model
            model = self.model.named_steps["model"]

            # Apply the same preprocessing used during training
            transformed_data = preprocessor.transform(input_df)

            # Convert sparse matrix if necessary
            if hasattr(transformed_data, "toarray"):
                transformed_data = transformed_data.toarray()

            # Get transformed feature names
            feature_names = preprocessor.get_feature_names_out()

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Calculate SHAP values
            shap_values = explainer.shap_values(transformed_data)

            # Handle binary classification output
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            # Create feature contribution table
            explanations = []

            for feature_name, shap_value in zip(
                feature_names,
                values
            ):
                explanations.append({
                    "feature": feature_name,
                    "shap_value": float(shap_value),
                    "absolute_value": abs(float(shap_value))
                })

            # Sort by importance
            explanations.sort(
                key=lambda x: x["absolute_value"],
                reverse=True
            )

            # Keep top 5 important features
            top_features = explanations[:5]

            # Add direction
            for item in top_features:

                if item["shap_value"] > 0:
                    item["direction"] = "Increases failure risk"

                elif item["shap_value"] < 0:
                    item["direction"] = "Decreases failure risk"

                else:
                    item["direction"] = "No significant contribution"

            return {
                "method": "SHAP",
                "features": top_features
            }

        except Exception as error:

            print("\nSHAP ERROR:")
            print(type(error).__name__)
            print(str(error))

            return {
                "method": "Fallback",
                "error": str(error),
                "features": []
            }