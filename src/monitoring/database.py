from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "data" / "feedback"
DB_PATH = DB_DIR / "predictions.db"


def get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def initialize_database() -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER,
            job TEXT,
            marital TEXT,
            education TEXT,
            default_status TEXT,
            balance REAL,
            housing TEXT,
            loan TEXT,
            contact TEXT,
            day INTEGER,
            month TEXT,
            campaign INTEGER,
            pdays INTEGER,
            previous INTEGER,
            poutcome TEXT,
            prediction INTEGER,
            prediction_label TEXT,
            prediction_probability REAL,
            actual_label TEXT
        )
        """
    )

    conn.commit()
    conn.close()