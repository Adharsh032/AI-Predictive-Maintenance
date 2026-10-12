from database.db import get_connection


def save_prediction(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    prediction,
    failure_probability,
    health_status=None,
    ai_method=None,
    ai_analysis=None,
):
    """Save a machine prediction and AI analysis to the database."""

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            product_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            prediction,
            failure_probability,
            health_status,
            ai_method,
            ai_analysis
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            product_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            prediction,
            failure_probability,
            health_status,
            ai_method,
            ai_analysis,
        ),
    )

    prediction_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return prediction_id


def get_prediction_by_id(prediction_id):
    """Retrieve one prediction by ID."""

    connection = get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE id = ?
        """,
        (prediction_id,),
    ).fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def get_all_predictions():
    """Retrieve all predictions, newest first."""

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY created_at DESC
        """
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_prediction_count():
    """Return the total number of predictions."""

    connection = get_connection()

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        """
    ).fetchone()[0]

    connection.close()

    return count

def get_dashboard_statistics():
    connection = get_connection()
    cursor = connection.cursor()

    # Total predictions
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
    """)
    total_predictions = cursor.fetchone()["total"]

    # Total failures
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE prediction = 1
    """)
    total_failures = cursor.fetchone()["total"]

    # Healthy machines
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE health_status = 'Healthy'
    """)
    healthy_machines = cursor.fetchone()["total"]

    # Warning machines
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE health_status = 'Warning'
    """)
    warning_machines = cursor.fetchone()["total"]

    # Critical machines
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE health_status = 'Critical'
    """)
    critical_machines = cursor.fetchone()["total"]

    # Prediction distribution
    cursor.execute("""
        SELECT prediction, COUNT(*) AS total
        FROM predictions
        GROUP BY prediction
    """)
    prediction_rows = cursor.fetchall()

    prediction_distribution = {
        "No Machine Failure": 0,
        "Machine Failure": 0
    }

    for row in prediction_rows:

        if row["prediction"] == 1:
            prediction_distribution["Machine Failure"] = row["total"]
        else:
            prediction_distribution["No Machine Failure"] = row["total"]

    # Health distribution
    health_distribution = {
        "Healthy": healthy_machines,
        "Warning": warning_machines,
        "Critical": critical_machines
    }

    # Recent predictions
    cursor.execute("""
        SELECT
            id,
            product_type,
            prediction,
            failure_probability,
            health_status,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """)

    recent_predictions = cursor.fetchall()

    connection.close()

    return {
        "total_predictions": total_predictions,
        "total_failures": total_failures,
        "healthy_machines": healthy_machines,
        "warning_machines": warning_machines,
        "critical_machines": critical_machines,
        "prediction_distribution": prediction_distribution,
        "health_distribution": health_distribution,
        "recent_predictions": recent_predictions
    }