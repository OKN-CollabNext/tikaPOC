import os
import sys  # <--- Add this import!
import logging
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque
from collections import deque

# Add the project root to sys.path so that logging_config.py can be imported.
# Assuming the structure is:
# /Users/deangladish/tikaPOC/logging_config.py
# /Users/deangladish/tikaPOC/src/services/topic_agent.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logging_config import setup_logging
setup_logging()  # This will configure logging and write to "app.log"

from openai import AzureOpenAI
from .topic_search import TopicSearcher
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

@dataclass
class AgentState:
    """Maintains agent's memory of excluded topics and recent queries."""
    excluded_topic_ids: Set[str] = field(default_factory=set)
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=5))

class TopicAgent:
    def __init__(self,
                 client: AzureOpenAI = None,
                 searcher: TopicSearcher = None):
        """
        Initialize the TopicAgent with required components.

        Args:
            client (AzureOpenAI, optional): An instance of AzureOpenAI. If not provided, one is created.
            searcher (TopicSearcher, optional): An instance of TopicSearcher. If not provided, one is created.
        """
        self.state = AgentState()
        self.searcher = searcher if searcher is not None else TopicSearcher()

        if client is not None:
            self.client = client
        else:
            self.client = AzureOpenAI(
                azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
                api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
                api_version="2024-12-01-preview"
            )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value

    def _rewrite_query(self) -> str:
        query_history = list(self.state.recent_queries)
        logger.info(f"Query history before rewrite: {query_history}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query rewriting assistant. Given a sequence of user queries, "
                    "rewrite them into a single, comprehensive query that captures the user's evolving intent. "
                    "Focus on creating a search-friendly query that works well with embedding-based search. "
                    "Return ONLY the rewritten query, nothing else."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Previous queries: {', '.join(query_history[:-1])}\n"
                    f"Current query: {query_history[-1]}\n"
                    "Rewrite these queries into a single, comprehensive query."
                )
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            logger.error(f"OpenAI API error in query rewriting: {e}")
            return query_history[-1]

    def _advanced_rewrite_query(self) -> str:
        query_history = list(self.state.recent_queries)
        logger.info(f"Query history before advanced rewrite: {query_history}")

        base_rewritten_query = self._rewrite_query()
        logger.info(f"Base rewritten query: {base_rewritten_query}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an advanced query rewriting assistant. Your job is to take a search query and "
                    "rewrite it to be more expressive and comprehensive by paraphrasing it and by "
                    "expanding key terms with appropriate synonyms. Focus on enhancing the query for an embedding-based search. "
                    "Return ONLY the rewritten query."
                )
            },
            {
                "role": "user",
                "content": f"Original query: {base_rewritten_query}\nPlease rewrite this query with synonyms and a paraphrased structure."
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
            advanced_query = response.choices[0].message.content.strip()
            logger.info(f"Advanced rewritten query: {advanced_query}")
            return advanced_query
        except Exception as e:
            logger.error(f"OpenAI API error in advanced query rewriting: {e}")
            return base_rewritten_query

    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        self.state.recent_queries.append(user_input)
        logger.info(f"Current query history: {list(self.state.recent_queries)}")
        logger.info(f"Number of queries in history: {len(self.state.recent_queries)}")

        should_rewrite = len(self.state.recent_queries) > 1
        logger.info(f"Should rewrite: {should_rewrite}")

        if should_rewrite:
            query_to_search = self._advanced_rewrite_query()
        else:
            query_to_search = user_input
        logger.info(f"Final search query: {query_to_search}")

        return self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids
        )

    def exclude_topics(self, topic_ids: List[str]) -> None:
        self.state.excluded_topic_ids.update(topic_ids)
        logger.info(f"Updated excluded topics: {self.state.excluded_topic_ids}")

    def reset_memory(self) -> None:
        self.state = AgentState()
        logger.info("Agent memory has been reset.")

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1
        }
