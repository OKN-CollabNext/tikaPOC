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


def fetch_patents_from_api() -> List[Dict[str, Any]]:
    """
    Fetch or load patents data from an external API or local JSON file.
    This is a placeholder that returns mock data.
    Replace with your real fetching logic.
    """
    # Example mock data
    mock_data = [
        {
            "id": "patent_abc",
            "title": "Method for Enhanced Battery Efficiency",
            "abstract": "Claims revolve around a novel approach to lithium-ion cathodes...",
            "inventor": ["John Inventor", "Mary Engineer"],
            "publication_date": "2022-09-01",
        },
        {
            "id": "patent_xyz",
            "title": "Quantum Dot Solar Cells",
            "abstract": "A technique for improving efficiency in quantum dot-based photovoltaics...",
            "inventor": ["Dr. Photon", "Jane Researcher"],
            "publication_date": "2021-12-15",
        },
    ]
    logger.info(f"Fetched {len(mock_data)} patents from API.")
    return mock_data


def insert_patents_to_db(patents_data: List[Dict[str, Any]]) -> None:
    """
    Embed each patent's text, then insert into DB.

    Args:
        patents_data: List of patent dictionaries.
    """
    logger.info("Starting insertion of patents into the database.")
    embedder = SciBertEmbedder()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for p in tqdm(patents_data, desc="Inserting patents"):
                text_for_embedding = (p.get("title", "") + " " + p.get("abstract", "")).strip()
                emb = embedder.get_embeddings_batch([text_for_embedding])[0]

                # Insert or upsert into the 'patents' table
                cur.execute(
                    """
                    INSERT INTO patents (id, title, abstract, inventor, publication_date, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET title = EXCLUDED.title,
                          abstract = EXCLUDED.abstract,
                          inventor = EXCLUDED.inventor,
                          publication_date = EXCLUDED.publication_date,
                          embedding = EXCLUDED.embedding
                    """,
                    (
                        p["id"],
                        p["title"],
                        p.get("abstract", ""),
                        p.get("inventor", []),
                        p.get("publication_date"),
                        emb.tolist(),
                    ),
                )
    conn.commit()
    logger.info("Completed insertion of patents into the database.")


def main():
    # 1) Fetch patents
    patents = fetch_patents_from_api()
    # 2) Insert patents into DB
    insert_patents_to_db(patents)


if __name__ == "__main__":
    main()
