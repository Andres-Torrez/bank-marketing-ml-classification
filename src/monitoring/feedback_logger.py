from __future__ import annotations

from datetime import datetime

from src.monitoring.database import get_connection, initialize_database


def log_prediction(user_input: dict, prediction: int, probability: float | None) -> None:
    """
    Save a new prediction event into SQLite database.
    """
    initialize_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_logs (
            timestamp,
            age,
            job,
            marital,
            education,
            default_status,
            balance,
            housing,
            loan,
            contact,
            day,
            month,
            campaign,
            pdays,
            previous,
            poutcome,
            prediction,
            prediction_label,
            prediction_probability,
            actual_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            user_input.get("age"),
            user_input.get("job"),
            user_input.get("marital"),
            user_input.get("education"),
            user_input.get("default"),
            user_input.get("balance"),
            user_input.get("housing"),
            user_input.get("loan"),
            user_input.get("contact"),
            user_input.get("day"),
            user_input.get("month"),
            user_input.get("campaign"),
            user_input.get("pdays"),
            user_input.get("previous"),
            user_input.get("poutcome"),
            int(prediction),
            "yes" if int(prediction) == 1 else "no",
            float(probability) if probability is not None else None,
            None,
        ),
    )

    conn.commit()
    conn.close()