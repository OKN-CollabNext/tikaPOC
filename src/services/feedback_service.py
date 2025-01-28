from typing import Optional
from .topic_search import get_db_connection
import logging
from datetime import datetime


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class FeedbackService:
    def save_feedback(
        self, user_id: Optional[str], rating: int, comment: Optional[str]
    ) -> None:
        """
        Store user feedback in the 'feedback' table.

        Args:
            user_id (Optional[str]): Identifier for the user (if available).
            rating (int): Rating between 1 and 5.
            comment (Optional[str]): Additional comments from the user.
        """
        if not (1 <= rating <= 5):
            logger.warning(f"Invalid rating received: {rating}. Must be between 1 and 5.")
            raise ValueError("Rating must be between 1 and 5.")

        timestamp = datetime.utcnow()
        logger.debug(f"Saving feedback: UserID={user_id}, Rating={rating}, Comment={comment}")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (user_id, rating, comment, timestamp)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, rating, comment, timestamp),
                )
        conn.commit()
        logger.info("User feedback saved successfully.")
