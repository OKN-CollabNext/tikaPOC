# File: /Users/deangladish/tikaPOC/src/services/feedback_service.py

from typing import Optional
from .topic_search import get_db_connection

class FeedbackService:
    def save_feedback(self, user_id: Optional[str], rating: int, comment: Optional[str]) -> None:
        """
        Store user feedback in the 'feedback' table.

        Args:
            user_id (Optional[str]): Identifier for the user (if available).
            rating (int): Rating between 1 and 5.
            comment (Optional[str]): Additional comments from the user.
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (user_id, rating, comment)
                    VALUES (%s, %s, %s)
                    """,
                    (user_id, rating, comment)
                )
            conn.commit()
