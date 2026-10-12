import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_connection


connection = get_connection()

cursor = connection.cursor()

cursor.execute("PRAGMA table_info(predictions)")

columns = cursor.fetchall()

print("\nPredictions table columns:\n")

for column in columns:
    print(
        f"{column['cid']}: "
        f"{column['name']} | "
        f"{column['type']}"
    )

connection.close()