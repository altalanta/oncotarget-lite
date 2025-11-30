"""Module for managing human-in-the-loop feedback."""

import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class FeedbackItem:
    gene: str
    prediction: float
    feedback: str # e.g., "Correct", "Incorrect", "Helpful", "Not Helpful"
    comment: str
    timestamp: datetime

class FeedbackStore:
    """Manages the storage and retrieval of user feedback."""
    
    def __init__(self, db_path: Path = Path("reports/feedback.db")):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and table."""
        self.db_path.parent.mkdir(exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gene TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    feedback TEXT NOT NULL,
                    comment TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

    def add_feedback(self, item: FeedbackItem):
        """Add a new feedback item to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feedback (gene, prediction, feedback, comment, timestamp) VALUES (?, ?, ?, ?, ?)",
                (item.gene, item.prediction, item.feedback, item.comment, item.timestamp.isoformat())
            )

    def get_all_feedback(self) -> List[FeedbackItem]:
        """Retrieve all feedback items from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT gene, prediction, feedback, comment, timestamp FROM feedback ORDER BY timestamp DESC")
            return [
                FeedbackItem(
                    gene=row[0],
                    prediction=row[1],
                    feedback=row[2],
                    comment=row[3],
                    timestamp=datetime.fromisoformat(row[4])
                ) for row in cursor.fetchall()
            ]

