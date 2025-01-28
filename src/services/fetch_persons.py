from typing import List, Dict, Any
from .topic_search import get_db_connection
from .populate_db import SciBertEmbedder
import logging
from tqdm import tqdm


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def insert_persons(persons: List[Dict[str, Any]]) -> None:
    """
    Insert or update person records in the database.

    Args:
        persons: List of person dictionaries containing id, name, affiliation, and profile_url.
    """
    logger.info("Starting insertion of persons into the database.")
    embedder = SciBertEmbedder()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for p in tqdm(persons, desc="Inserting persons"):
                # Insert or upsert into the 'persons' table
                cur.execute(
                    """
                    INSERT INTO persons (id, name, affiliation, profile_url)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET name = EXCLUDED.name,
                          affiliation = EXCLUDED.affiliation,
                          profile_url = EXCLUDED.profile_url
                    """,
                    (
                        p["id"],
                        p["name"],
                        p.get("affiliation"),
                        p.get("profile_url"),
                    ),
                )
    conn.commit()
    logger.info("Completed insertion of persons into the database.")


def main():
    # Suppose you have some data source or just mock data:
    mock_persons = [
        {
            "id": "person_001",
            "name": "Jane Researcher",
            "affiliation": "Example University",
            "profile_url": "https://example.edu/~jane",
        },
        {
            "id": "person_002",
            "name": "John Innovator",
            "affiliation": "Tech Corp",
            "profile_url": "https://techcorp.com/john",
        },
    ]
    insert_persons(mock_persons)


if __name__ == "__main__":
    main()
