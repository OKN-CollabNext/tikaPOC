# File: /Users/deangladish/tikaPOC/src/services/fetch_patents.py

import requests
import time
import json
from typing import List, Dict, Any
from tqdm import tqdm
from .topic_search import get_db_connection
from .populate_db import SciBertEmbedder

def fetch_patents_from_api() -> List[Dict[str, Any]]:
    """
    Placeholder function to fetch or load patents data.
    """
    # Example mock data
    mock_data = [
        {
            "id": "patent_abc",
            "title": "Method for Enhanced Battery Efficiency",
            "abstract": "Claims revolve around a novel approach to lithium-ion cathodes...",
            "inventor": ["John Inventor", "Mary Engineer"],
            "publication_date": "2022-09-01"
        },
        {
            "id": "patent_xyz",
            "title": "Quantum Dot Solar Cells",
            "abstract": "A technique for improving efficiency in quantum dot-based photovoltaics...",
            "inventor": ["Dr. Photon", "Jane Researcher"],
            "publication_date": "2021-12-15"
        }
    ]
    return mock_data

def insert_patents_to_db(patents_data: List[Dict[str, Any]]) -> None:
    embedder = SciBertEmbedder()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for p in tqdm(patents_data, desc="Inserting patents"):
                text_for_embedding = (p.get("title","") + " " + p.get("abstract","")).strip()
                emb = embedder.get_embeddings_batch([text_for_embedding])[0]

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
                        p.get("abstract",""),
                        p.get("inventor", []),
                        p.get("publication_date"),
                        emb.tolist()
                    )
                )

        conn.commit()

def main():
    patents = fetch_patents_from_api()
    insert_patents_to_db(patents)

if __name__ == "__main__":
    main()
