from typing import List, Dict, Any, Set
import numpy as np
import psycopg2
from psycopg2.extensions import connection
import torch
from transformers import AutoTokenizer, AutoModel
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

class TopicSearcher:
    def __init__(self):
        """Initialize the searcher with the SciBERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model.eval()  # Set to evaluation mode

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding.squeeze()

    def search_topics(self, query: str, excluded_topic_ids: Set[str], n_topics: int = 5) -> List[Dict[str, Any]]:
        documents = self.simulate_document_retrieval()
        query_embedding = self.get_embedding(query)
        doc_embeddings = np.array([self.get_embedding(doc) for doc in documents])

        cosine_similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(-cosine_similarities)[:n_topics]
        top_topics = [
            {
                'id': str(i),
                'display_name': documents[i],
                'description': f'Description for {documents[i]}'
            } for i in top_indices if str(i) not in excluded_topic_ids
        ]

        return top_topics

    def simulate_document_retrieval(self) -> List[str]:
        """
        Simulate the retrieval of documents. This should ideally fetch from a database or another data source.
        """
        return [
            "Topic 1: Advances in Quantum Computing",
            "Topic 2: Neural Networks and Deep Learning",
            "Topic 3: AI in Healthcare",
            "Topic 4: Machine Learning in Financial Markets",
            "Topic 5: Natural Language Processing",
            "Topic 6: Sustainability through AI",
            "Topic 7: Robotics and Automation"
        ]

def get_db_connection() -> connection:
    """
    Create a database connection using secrets from Azure Key Vault.
    """
    host = secret_client.get_secret("DB-HOST").value
    db_name = secret_client.get_secret("DATABASE-NAME").value
    user = secret_client.get_secret("DB-USER").value
    password = secret_client.get_secret("DB-PASSWORD").value
    port = secret_client.get_secret("DB-PORT").value

    try:
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
    except psycopg2.Error as e:
        print(f"PostgreSQL Error: {e.pgcode} - {e.pgerror}")
        raise
    except Exception as e:
        print(f"Connection error details: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Failed to connect to database: {str(e)}")
