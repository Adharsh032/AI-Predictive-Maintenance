import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH


def get_connection():
    """Create and return a SQLite database connection."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():
    """Create and update the predictions table."""

    connection = get_connection()

    cursor = connection.cursor()

    # Create the original table if it does not exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            product_type TEXT NOT NULL,

            air_temperature REAL NOT NULL,
            process_temperature REAL NOT NULL,
            rotational_speed REAL NOT NULL,
            torque REAL NOT NULL,
            tool_wear REAL NOT NULL,

            prediction INTEGER NOT NULL,
            failure_probability REAL NOT NULL,

            health_status TEXT,
            ai_method TEXT,
            ai_analysis TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Add new columns to an existing database if they are missing
    cursor.execute(
        "PRAGMA table_info(predictions)"
    )

    existing_columns = {
        row["name"]
        for row in cursor.fetchall()
    }

    if "health_status" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN health_status TEXT
            """
        )

    if "ai_method" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN ai_method TEXT
            """
        )

    if "ai_analysis" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN ai_analysis TEXT
            """
        )

    connection.commit()
    connection.close()


if __name__ == "__main__":
    init_database()

    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    print("\nDatabase created/verified at:")
    print(DATABASE_PATH)
    print("\nPredictions table is ready.")