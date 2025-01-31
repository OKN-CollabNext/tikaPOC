import os
import logging
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque
from collections import deque

from openai import AzureOpenAI
from .topic_search import TopicSearcher
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Configure logging
logging.basicConfig(level=logging.INFO)
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

        # Use dependency injection for the OpenAI client; otherwise, create one using Key Vault secrets.
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
        """
        Rewrite the query using context from recent queries.
        Only called when there are multiple queries in history.

        Returns:
            str: The rewritten query.
        """
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
            # Fallback to using the latest query if rewriting fails
            return query_history[-1]

    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.

        Args:
            user_input (str): The user's query string.

        Returns:
            List[Dict[str, Any]]: List of topic dictionaries containing id, display_name, and description.
        """
        # Add the new query to the history
        self.state.recent_queries.append(user_input)
        logger.info(f"Current query history: {list(self.state.recent_queries)}")
        logger.info(f"Number of queries in history: {len(self.state.recent_queries)}")

        # Decide whether to rewrite the query
        should_rewrite = len(self.state.recent_queries) > 1
        logger.info(f"Should rewrite: {should_rewrite}")

        # Use the rewritten query if appropriate
        query_to_search = self._rewrite_query() if should_rewrite else user_input
        logger.info(f"Final search query: {query_to_search}")

        # Return topics matching the search query, excluding topics already in state
        return self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids
        )

    def exclude_topics(self, topic_ids: List[str]) -> None:
        """
        Add topics to the excluded set.

        Args:
            topic_ids (List[str]): List of topic IDs to exclude.
        """
        self.state.excluded_topic_ids.update(topic_ids)
        logger.info(f"Updated excluded topics: {self.state.excluded_topic_ids}")

    def reset_memory(self) -> None:
        """Reset the agent's memory (both excluded topics and query history)."""
        self.state = AgentState()
        logger.info("Agent memory has been reset.")

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current agent state for debugging.

        Returns:
            Dict[str, Any]: A summary containing the count of excluded topics, recent queries, and whether context exists.
        """
        return {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1
        }
