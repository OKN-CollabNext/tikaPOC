import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque
from collections import deque
from openai import AzureOpenAI
from .topic_search import TopicSearcher
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

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
                model=self.deployment,  # Use deployment from Key Vault
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            print(f"OpenAI API error in query rewriting: {str(e)}")
            # Fallback to using the latest query if rewriting fails
            return query_history[-1]

    def ask_for_clarification(self, ambiguous_topics: List[Dict[str, Any]]) -> str:
        """
        Ask the user a clarifying question when multiple ambiguous topics are found.
        """
        clarification_prompt = (
            "I found several topics that match your query:\n" +
            "\n".join([f"- {t['display_name']}: {t['description']}" for t in ambiguous_topics]) +
            "\nCould you please clarify which aspect you are most interested in?"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a clarifying assistant helping a user refine their research topic query."
            },
            {"role": "user", "content": clarification_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
            clarification = response.choices[0].message.content.strip()
            print(f"Clarification generated: {clarification}")
            return clarification
        except Exception as e:
            print(f"OpenAI API error during clarification: {str(e)}")
            return "Could you please clarify your research topic further?"

    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.
        If ambiguous results are found, return a clarification prompt.
        """
        # Add new query to history
        self.state.recent_queries.append(user_input)
        print(f"Current query history: {list(self.state.recent_queries)}")
        print(f"Number of queries in history: {len(self.state.recent_queries)}")

        # Determine if we should rewrite the query
        should_rewrite = len(self.state.recent_queries) > 1
        print(f"Should rewrite: {should_rewrite}")
        query_to_search = self._rewrite_query() if should_rewrite else user_input
        print(f"Final search query: {query_to_search}")

        # Search for topics using excluded topics list
        results = self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids
        )

        # If the top two results are very close, ask for clarification.
        if len(results) > 1 and abs(results[0]["score"] - results[1]["score"]) < 0.1:
            clarification = self.ask_for_clarification(results[:3])
            # Return a special result indicating that clarification is needed.
            return [{"clarification": clarification}]

        return results

    def exclude_topics(self, topic_ids: List[str]) -> None:
        """Add topics to the excluded set."""
        self.state.excluded_topic_ids.update(topic_ids)

    def reset_memory(self) -> None:
        """Reset the agent's memory (both excluded topics and query history)."""
        self.state = AgentState()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current agent state for debugging."""
        return {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1
        }
