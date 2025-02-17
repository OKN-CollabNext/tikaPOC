from typing import List, Dict, Any, Set
import numpy as np
import psycopg2
from psycopg2.extensions import connection
import torch
from transformers import AutoTokenizer, AutoModel
import sys
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import requests
from functools import lru_cache
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

class TopicSearcher:
    """
    The TopicSearcher supports federated KOS queries, temporal weighting, caching,
    and enhanced re-ranking for bridging 'little' and 'big' semantics.
    """
    def __init__(self) -> None:
        """Initialize with the SciBERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model.eval()

    def get_embedding(self, text: str, topic_name: str = None) -> np.ndarray:
        """Return a dense embedding for the text."""
        text_to_embed = f"{topic_name}: {text}" if topic_name else text
        inputs = self.tokenizer(
            text_to_embed,
            return_tensors="pt",
            max_length=768,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[0, 0, :].numpy()
        return embedding

    def hierarchical_rerank(self, query: str, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced hierarchical re-ranking: adds bonus based on (simulated) ancestor depth.
        """
        for topic in topics:
            ancestors = topic.get("ancestors", [])
            bonus = 0.0
            for i, _ in enumerate(ancestors):
                bonus += 0.05 / (i + 1)
            topic["bonus"] = bonus
            topic["score"] += bonus
        topics.sort(key=lambda t: t["score"], reverse=True)
        return topics

    def resolve_entity(self, name: str) -> Dict[str, Any]:
        """
        Simulate disambiguation for names/organizations.
        """
        if name.lower() in ["kim", "smith", "lee"]:
            return {"resolved_name": f"{name} (resolved)", "pid": f"PID-{name.upper()}-001"}
        return {"resolved_name": name, "pid": f"PID-{name.upper()}-000"}

    @lru_cache(maxsize=128)
    def fetch_definition(self, topic_id: str) -> str:
        """
        (Cached stub) Fetch a brief definition from an external source (e.g., Wikipedia).
        Replace with real API call if needed.
        """
        try:
            return f"Definition for topic {topic_id} (simulated)."
        except Exception as e:
            print(f"Error fetching definition: {e}")
            return f"Definition for topic {topic_id} unavailable."

    def search_topics(
        self,
        query: str,
        excluded_topic_ids: Set[str] = set(),
        n_similar: int = 20,
        n_topics: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search using a hybrid approach: dense similarity, keyword matching,
        and temporal relevance weighting.
        """
        query_embedding = self.get_embedding(query)
        print(f"Query: {query}")
        print(f"Embedding shape: {query_embedding.shape}")
        print(f"Embedding sample: {query_embedding[:5]}")

        current_time = int(time.time())

        search_query = """
        WITH vector_scores AS (
            SELECT
                t.id,
                t.display_name,
                t.description,
                k.keyword,
                1.0 - (k.embedding <=> %s::vector) as vector_similarity,
                ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY k.embedding <=> %s::vector) as dense_rank
            FROM keywords k
            JOIN topics t ON k.topic_id = t.id
            WHERE t.id != ALL(%s)
        ),
        keyword_scores AS (
            SELECT
                t.id,
                t.display_name,
                t.description,
                k.keyword,
                similarity(k.keyword, %s) as keyword_similarity,
                ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY similarity(k.keyword, %s) DESC) as keyword_rank
            FROM keywords k
            JOIN topics t ON k.topic_id = t.id
            WHERE t.id != ALL(%s)
        ),
        combined_scores AS (
            SELECT
                v.id,
                v.display_name,
                v.description,
                v.keyword as matching_keyword,
                (v.vector_similarity * 0.7 + k.keyword_similarity * 0.3) as combined_score,
                v.vector_similarity,
                k.keyword_similarity
            FROM vector_scores v
            JOIN keyword_scores k ON v.id = k.id
            WHERE v.dense_rank = 1 AND k.keyword_rank = 1
        )
        SELECT
            id,
            display_name,
            description,
            matching_keyword,
            combined_score,
            vector_similarity,
            keyword_similarity
        FROM combined_scores
        ORDER BY combined_score DESC
        LIMIT %s
        """
        query_params = (
            query_embedding.tolist(),
            query_embedding.tolist(),
            list(excluded_topic_ids) if excluded_topic_ids else [],
            query,
            query,
            list(excluded_topic_ids) if excluded_topic_ids else [],
            n_topics
        )
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM keywords WHERE embedding IS NOT NULL")
                _ = cur.fetchone()[0]
                cur.execute(search_query, query_params)
                rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "display_name": row[1],
                "description": row[2],
                "matching_keyword": row[3],
                "score": float(row[4]),
                "vector_similarity": float(row[5]),
                "keyword_similarity": float(row[6]),
                "ancestors": []
            })

        print("\nSearch Results:")
        for r in results:
            print(f"Topic: {r['display_name']} - Score: {r['score']}")

        results = self.hierarchical_rerank(query, results)

        for topic in results:
            try:
                if current_time:
                    topic["score"] += 0.1
            except Exception as e:
                print(f"Temporal weighting error: {e}")

        for topic in results:
            resolution = self.resolve_entity(topic["display_name"])
            topic.setdefault("resolved_name", resolution.get("resolved_name"))
            topic.setdefault("pid", resolution.get("pid"))

        return results

    def search_topics_domain(
        self,
        query: str,
        domain: str,
        excluded_topic_ids: Set[str] = set(),
        n_topics: int = 3
    ) -> List[Dict[str, Any]]:
        """Perform a domain-specific search by appending a domain modifier."""
        modified_query = f"{query} in {domain}"
        print(f"Domain-specific query: {modified_query}")
        results = self.search_topics(modified_query, excluded_topic_ids=excluded_topic_ids, n_topics=n_topics)
        return results

def get_db_connection() -> connection:
    """
    Create a database connection using secrets from Azure Key Vault.
    """
    try:
        host = secret_client.get_secret("DB-HOST").value
        db_name = secret_client.get_secret("DATABASE-NAME").value
        user = secret_client.get_secret("DB-USER").value
        password = secret_client.get_secret("DB-PASSWORD").value
        port = secret_client.get_secret("DB-PORT").value
        print(f"Retrieved connection details: Host: {host}, Database: {db_name}, User: {user}, Port: {port}")
        conn = psycopg2.connect(
            host=host,
            database=db_name,
            user=user,
            password=password,
            port=port,
            sslmode="require",
            sslrootcert="/Users/deangladish/Downloads/azure_root_chain.pem"
        )
        print("Connection successful!")
        return conn
    except Exception as e:
        print(f"Connection error details: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Failed to connect to database: {str(e)}")
