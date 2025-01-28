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


def fetch_conferences_from_api() -> List[Dict[str, Any]]:
    """
    Fetch or load conferences data from an external API or local JSON file.
    This is a placeholder that returns mock data.
    Replace with your real fetching logic.
    """
    mock_data = [
        {
            "id": "conf_2024_ai",
            "name": "International Conference on AI Research",
            "location": "Paris, France",
            "start_date": "2024-07-10",
            "end_date": "2024-07-14",
            "description": "Leading conference on AI, focusing on large language models...",
        },
        {
            "id": "conf_2025_quantum",
            "name": "Quantum Computing Summit",
            "location": "Boston, MA",
            "start_date": "2025-03-01",
            "end_date": "2025-03-03",
            "description": "Annual event for quantum computing research...",
        },
    ]
    logger.info(f"Fetched {len(mock_data)} conferences from API.")
    return mock_data


def insert_conferences_to_db(conferences_data: List[Dict[str, Any]]) -> None:
    """
    Embed each conference's text, then insert into DB.

    Args:
        conferences_data: List of conference dictionaries.
    """
    logger.info("Starting insertion of conferences into the database.")
    embedder = SciBertEmbedder()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for c in tqdm(conferences_data, desc="Inserting conferences"):
                text_for_embedding = (c.get("name", "") + " " + c.get("description", "")).strip()
                emb = embedder.get_embeddings_batch([text_for_embedding])[0]

                # Insert or upsert into the 'conferences' table
                cur.execute(
                    """
                    INSERT INTO conferences (id, name, location, start_date, end_date, description, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET name = EXCLUDED.name,
                          location = EXCLUDED.location,
                          start_date = EXCLUDED.start_date,
                          end_date = EXCLUDED.end_date,
                          description = EXCLUDED.description,
                          embedding = EXCLUDED.embedding
                    """,
                    (
                        c["id"],
                        c["name"],
                        c.get("location"),
                        c.get("start_date"),
                        c.get("end_date"),
                        c.get("description", ""),
                        emb.tolist(),
                    ),
                )
    conn.commit()
    logger.info("Completed insertion of conferences into the database.")


def main():
    # 1) Fetch conferences
    conferences = fetch_conferences_from_api()
    # 2) Insert conferences into DB
    insert_conferences_to_db(conferences)


if __name__ == "__main__":
    main()
