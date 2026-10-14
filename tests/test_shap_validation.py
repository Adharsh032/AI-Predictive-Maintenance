
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from config import MODEL_PATH
from services.explanation_service import ExplanationService


service = ExplanationService()
pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(model)

test_cases = [
    {
        "name": "Normal-looking input",
        "input": {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100,
        },
    },
    {
        "name": "Higher-risk test input",
        "input": {
            "Type": "L",
            "Air temperature [K]": 303.0,
            "Process temperature [K]": 313.0,
            "Rotational speed [rpm]": 1380,
            "Torque [Nm]": 55.0,
            "Tool wear [min]": 240,
        },
    },
]


for case in test_cases:
    print("\n" + "=" * 60)
    print(case["name"])
    print("=" * 60)

    input_df = pd.DataFrame([case["input"]])

    # Get prediction and probability from the complete pipeline.
    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0, 1])

    print("Predicted class:", prediction)
    print(f"Failure probability: {probability:.6f}")

    # Apply the exact fitted preprocessing used by the model.
    transformed = preprocessor.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    # Obtain SHAP values and base value.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        explanation = explainer(transformed)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    # Normalize the binary-classification output shape.
    if values.ndim == 3:
        # Shape can be samples x features x classes.
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    base_value = float(base_values.reshape(-1)[0])
    shap_sum = float(np.sum(values))
    reconstructed_raw_output = base_value + shap_sum

    # For the usual LightGBM TreeExplainer raw output,
    # the reconstructed output is a log-odds score.
    predicted_probability_from_shap = (
        1.0 / (1.0 + np.exp(-reconstructed_raw_output))
    )

    print(f"SHAP base value: {base_value:.6f}")
    print(f"Sum of SHAP values: {shap_sum:.6f}")
    print(f"Base + SHAP sum: {reconstructed_raw_output:.6f}")
    print(
        "Probability from reconstructed output: "
        f"{predicted_probability_from_shap:.6f}"
    )
    print(
        "Difference from pipeline probability: "
        f"{abs(predicted_probability_from_shap - probability):.8f}"
    )

    # Display the application's explanation too.
    result = service.explain_prediction(case["input"])
    print("Application explanation method:", result["method"])

    if result["method"] != "SHAP":
        print("Explanation error:", result.get("error"))
        continue

    print("Top feature contributions:")
    feature_names = preprocessor.get_feature_names_out()

    ranked = sorted(
        zip(feature_names, values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    for feature_name, value in ranked[:5]:
        direction = (
            "Increases failure risk"
            if value > 0
            else "Decreases failure risk"
            if value < 0
            else "No contribution"
        )
        print(f"{feature_name}: {float(value):.6f} ({direction})")
