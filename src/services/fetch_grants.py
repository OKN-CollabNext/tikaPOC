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


def fetch_grants_from_api() -> List[Dict[str, Any]]:
    """
    Fetch or load a list of grants from some external API or local JSON file.
    This is just a placeholder that returns mock data.
    Replace with your real fetching logic.
    """
    # Example: returning fake data
    mock_data = [
        {
            "id": "grant_001",
            "title": "Deep Learning for Protein Folding",
            "abstract": "This grant focuses on neural network architectures...",
            "investigators": ["Alice Smith", "Bob Jones"],
            "start_date": "2024-01-01",
            "end_date": "2026-12-31",
        },
        {
            "id": "grant_002",
            "title": "Quantum Computing Approaches to Cryptography",
            "abstract": "Research on post-quantum cryptographic algorithms...",
            "investigators": ["Carol White"],
            "start_date": "2023-06-01",
            "end_date": "2025-05-31",
        },
    ]
    logger.info(f"Fetched {len(mock_data)} grants from API.")
    return mock_data


def insert_grants_to_db(grants_data: List[Dict[str, Any]]) -> None:
    """
    Embed each grant's text, then insert into DB.

    Args:
        grants_data: List of grant dictionaries.
    """
    logger.info("Starting insertion of grants into the database.")
    embedder = SciBertEmbedder()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for g in tqdm(grants_data, desc="Inserting grants"):
                # Get embedding for the 'title + abstract'
                text_for_embedding = (g.get("title", "") + " " + g.get("abstract", "")).strip()
                emb = embedder.get_embeddings_batch([text_for_embedding])[0]

                # Insert or upsert into the 'grants' table
                cur.execute(
                    """
                    INSERT INTO grants (id, title, abstract, investigators, start_date, end_date, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET title = EXCLUDED.title,
                          abstract = EXCLUDED.abstract,
                          investigators = EXCLUDED.investigators,
                          start_date = EXCLUDED.start_date,
                          end_date = EXCLUDED.end_date,
                          embedding = EXCLUDED.embedding
                    """,
                    (
                        g["id"],
                        g["title"],
                        g.get("abstract", ""),
                        g.get("investigators", []),
                        g.get("start_date"),
                        g.get("end_date"),
                        emb.tolist(),
                    ),
                )
    conn.commit()
    logger.info("Completed insertion of grants into the database.")


def main():
    # 1) Fetch grants
    grants = fetch_grants_from_api()
    # 2) Insert grants into DB
    insert_grants_to_db(grants)


if __name__ == "__main__":
    main()
