import json
from pathlib import Path


class KnowledgeBase:

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent

        self.file_path = (
            project_root
            / "data"
            / "processed"
            / "maintenance_knowledge.json"
        )

        self.knowledge = self._load_knowledge()

    def _load_knowledge(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def retrieve(self, input_data, prediction):
        relevant = []

        tool_wear = input_data["Tool wear [min]"]
        torque = input_data["Torque [Nm]"]
        rotational_speed = input_data["Rotational speed [rpm]"]
        air_temperature = input_data["Air temperature [K]"]
        process_temperature = input_data["Process temperature [K]"]

        temperature_difference = (
            process_temperature - air_temperature
        )

        # High tool wear
        if tool_wear >= 150:
            relevant.append(
                self._find_condition("High tool wear")
            )

        # High torque
        if torque >= 50:
            relevant.append(
                self._find_condition("High torque")
            )

        # Abnormal rotational speed
        if rotational_speed < 1200 or rotational_speed > 2000:
            relevant.append(
                self._find_condition("Abnormal rotational speed")
            )

        # High process temperature
        if process_temperature >= 310:
            relevant.append(
                self._find_condition("High process temperature")
            )

        # High air temperature
        if air_temperature >= 300:
            relevant.append(
                self._find_condition("High air temperature")
            )

        # Temperature imbalance
        if temperature_difference >= 12:
            relevant.append(
                self._find_condition("Temperature imbalance")
            )

        # Failure prediction
        if prediction == 1:
            relevant.append(
                self._find_condition("General machine inspection")
            )

        # Remove None values
        relevant = [
            item for item in relevant
            if item is not None
        ]

        return relevant

    def _find_condition(self, condition):
        for item in self.knowledge:
            if item["condition"] == condition:
                return item

        return None