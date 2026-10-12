import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from database.db import init_database
from database.models import (
    save_prediction,
    get_prediction_by_id,
    get_all_predictions,
    get_prediction_count,
)


init_database()

print("=" * 60)
print("DATABASE MODEL TEST")
print("=" * 60)


prediction_id = save_prediction(
    product_type="M",
    air_temperature=300.0,
    process_temperature=310.0,
    rotational_speed=1500.0,
    torque=40.0,
    tool_wear=100.0,
    prediction=0,
    failure_probability=0.0001,
)

print(f"\nSaved prediction ID: {prediction_id}")


prediction = get_prediction_by_id(prediction_id)

print("\nRetrieved prediction:")
print(prediction)


count = get_prediction_count()

print(f"\nTotal predictions: {count}")


all_predictions = get_all_predictions()

print("\nAll predictions:")

for item in all_predictions:
    print(item)