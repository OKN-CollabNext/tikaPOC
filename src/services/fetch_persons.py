# File: fetch_persons.py
from .topic_search import get_db_connection

def insert_persons(persons: List[Dict[str, Any]]) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for p in persons:
                cur.execute(
                    """
                    INSERT INTO persons (id, name, affiliation, profile_url)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET name = EXCLUDED.name,
                          affiliation = EXCLUDED.affiliation,
                          profile_url = EXCLUDED.profile_url
                    """,
                    (p["id"], p["name"], p.get("affiliation"), p.get("profile_url"))
                )
        conn.commit()

def main():
    # Suppose you have some data source or just mock data:
    mock_persons = [
        {
            "id": "person_001",
            "name": "Jane Researcher",
            "affiliation": "Example University",
            "profile_url": "https://example.edu/~jane"
        }
    ]
    insert_persons(mock_persons)

if __name__ == "__main__":
    main()
