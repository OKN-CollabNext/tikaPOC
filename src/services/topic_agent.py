import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque, Optional
from collections import deque
from openai import AzureOpenAI
from .topic_search import TopicSearcher
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# New imports for ML classification and knowledge graph
from .ml_topic_classifier import classify_text  # (stub: implement your classifier here)
from .knowledge_graph import KnowledgeGraph  # (stub: implement a knowledge graph module)

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

@dataclass
class AgentState:
    """Maintains agent's memory of excluded topics and recent queries."""
    excluded_topic_ids: Set[str] = field(default_factory=set)
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    # Store last location (if provided) for geospatial recommendations
    last_location: Optional[Dict[str, float]] = None  # e.g. {"lat": ..., "lon": ...}

class TopicAgent:
    def __init__(self):
        """Initialize the agent with necessary components."""
        self.state = AgentState()
        self.searcher = TopicSearcher()
        self.client = AzureOpenAI(
            azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value
        # Initialize a knowledge graph to later incorporate relationships between topics and researchers
        self.kg = KnowledgeGraph()

    def _rewrite_query(self) -> str:
        """
        Rewrite the query using context from recent queries.
        Only called when there are multiple queries in history.
        """
        query_history = list(self.state.recent_queries)
        print(f"Query history before rewrite: {query_history}")

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
                    f"Current query: {query_history[-1]}\n\n"
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
            print(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            print(f"OpenAI API error in query rewriting: {str(e)}")
            return query_history[-1]

    def process_query(self, user_input: str, location: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.

        Args:
            user_input: The user's query string.
            location: Optional geospatial location (latitude and longitude) for recommendations.

        Returns:
            List of topic dictionaries containing id, display_name, description, etc.
        """
        # Save geospatial context if provided
        if location:
            self.state.last_location = location

        # Add new query to history
        self.state.recent_queries.append(user_input)

        # Debug prints
        print(f"Current query history: {list(self.state.recent_queries)}")
        print(f"Number of queries in history: {len(self.state.recent_queries)}")

        # Determine if we should rewrite the query
        should_rewrite = len(self.state.recent_queries) > 1
        print(f"Should rewrite: {should_rewrite}")

        # Get search query (rewrite if needed)
        query_to_search = self._rewrite_query() if should_rewrite else user_input
        print(f"Final search query: {query_to_search}")

        # --- New: Machine Learning–Based Topic Classification ---
        # Optionally, classify the input to detect broader research themes.
        classified_theme = classify_text(query_to_search)
        print(f"Classified theme: {classified_theme}")
        # You might use the classified theme to modify the query or enrich the search.
        enriched_query = f"{query_to_search} ({classified_theme})"

        # Search using the enriched query and excluded topics
        results = self.searcher.search_topics(
            query=enriched_query,
            excluded_topic_ids=self.state.excluded_topic_ids
        )

        # --- New: Geospatial re-ranking (stub) ---
        if location:
            results = self._rerank_by_geospatial(results, location)

        # --- New: Update the knowledge graph ---
        self.kg.update_with_query(query_to_search, results)

        return results

    def _rerank_by_geospatial(self, topics: List[Dict[str, Any]], location: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Re-rank topics based on regional proximity.
        (In practice, you would have geospatial metadata in the topics or linked researcher profiles.)
        Here we provide a stub that would re-sort topics.
        """
        print(f"Re-ranking {len(topics)} topics based on location: {location}")
        # Stub: simply return topics unchanged
        # (You could integrate a distance metric here if topics/researchers have lat/lon fields.)
        return topics

    def exclude_topics(self, topic_ids: List[str]) -> None:
        """Add topics to the excluded set."""
        self.state.excluded_topic_ids.update(topic_ids)

    def reset_memory(self) -> None:
        """Reset the agent's memory (both excluded topics and query history)."""
        self.state = AgentState()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current agent state for debugging."""
        return {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1,
            "last_location": self.state.last_location,
        }
