import os
import logging
import json
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque, Optional, Callable
from collections import deque
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_search import TopicSearcher
from utils.metrics import record_query

# Load environment variables from .env if available
load_dotenv()

# Set up structured JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        record_dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "funcName": record.funcName,
            "message": record.getMessage()
        }
        return json.dumps(record_dict)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Set up Key Vault client using environment variable (or default)
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

# Configuration constants (can be overridden via environment variables)
MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))
API_TEMPERATURE = float(os.getenv("API_TEMPERATURE", "0.3"))
KOS_HINT = os.getenv("KOS_HINT", "Candidate topics suggest areas such as Environmental Science, Waste Management, and Biomaterials.")

@dataclass
class AgentState:
    """Maintains agent's memory of excluded topics, recent queries, and clarification rounds."""
    excluded_topic_ids: Set[str] = field(default_factory=set)
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    clarification_round: int = 0

class TopicAgent:
    def __init__(self, feedback_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the agent with necessary components.
        :param feedback_callback: Optional callback function to record user feedback for clarifications.
        """
        self.state = AgentState()
        self.searcher = TopicSearcher()
        self.client = AzureOpenAI(
            azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value
        self.feedback_callback = feedback_callback
        logger.debug("Initialized TopicAgent with deployment: %s", self.deployment)

    def _integrate_hierarchical_metadata(self, query: str) -> str:
        """
        Retrieve and append hierarchical metadata from the KOS.
        """
        metadata = f" {KOS_HINT}"
        logger.debug("Appending hierarchical metadata to query: %s", metadata)
        return query + metadata

    def _rewrite_query(self) -> str:
        """
        Rewrite the query using context from recent queries and include hints from the knowledge organization system.
        """
        query_history = list(self.state.recent_queries)
        logger.debug("Query history before rewrite: %s", query_history)
        combined_query = f"Previous queries: {', '.join(query_history[:-1])}; Current query: {query_history[-1]}"
        combined_query = self._integrate_hierarchical_metadata(combined_query)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query rewriting assistant. Given a sequence of user queries and context about candidate research topic areas, "
                    "rewrite them into a single, comprehensive query that captures the user's evolving intent. "
                    "If the query is too broad, add suggestions to make it more specific. "
                    "Return ONLY the rewritten query."
                )
            },
            {
                "role": "user",
                "content": combined_query
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=API_TEMPERATURE,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            logger.debug("Rewritten query: %s", rewritten)
            return rewritten
        except Exception as e:
            logger.error("OpenAI API error in query rewriting: %s", str(e))
            return query_history[-1]

    def ask_for_clarification(self, ambiguous_topics: List[Dict[str, Any]]) -> str:
        """
        Ask the user a clarifying question when multiple ambiguous topics are found.
        """
        clarification_prompt = (
            "I found several topics that match your query:\n" +
            "\n".join([f"- {t['display_name']}: {t['description']}" for t in ambiguous_topics]) +
            "\nIt appears that your query may be too broad. " +
            "Could you please clarify which specific aspect you are most interested in? "
            "For example, specify the domain (e.g., 'renewable energy technologies'), a particular focus (e.g., 'recent advances in battery technology'), or any context that might narrow the search."
        )
        messages = [
            {
                "role": "system",
                "content": "You are a clarifying assistant helping a user refine their research topic query."
            },
            {
                "role": "user",
                "content": clarification_prompt
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=API_TEMPERATURE,
                max_tokens=150
            )
            clarification = response.choices[0].message.content.strip()
            logger.debug("Clarification generated: %s", clarification)
            if self.feedback_callback:
                self.feedback_callback(clarification)
            return clarification
        except Exception as e:
            logger.error("OpenAI API error during clarification: %s", str(e))
            return ("Your query seems ambiguous. Please refine your query by including more specific details, "
                    "such as a particular sub-field or targeted questions.")

    @record_query
    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.
        Implements an iterative Socratic approach for clarifying ambiguous queries.
        """
        self.state.recent_queries.append(user_input)
        logger.debug("Current query history: %s", list(self.state.recent_queries))
        logger.debug("Number of queries in history: %d", len(self.state.recent_queries))

        should_rewrite = len(self.state.recent_queries) > 1
        logger.debug("Should rewrite: %s", should_rewrite)
        query_to_search = self._rewrite_query() if should_rewrite else user_input
        logger.debug("Final search query: %s", query_to_search)

        results = self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids
        )

        if len(results) > 1 and abs(results[0]["score"] - results[1]["score"]) < 0.1:
            if self.state.clarification_round < MAX_CLARIFICATION_ROUNDS:
                self.state.clarification_round += 1
                clarification = self.ask_for_clarification(results[:3])
                logger.debug("Clarification round %d result: %s", self.state.clarification_round, clarification)
                return self.process_query(clarification)
            else:
                fallback = {
                    "clarification": (
                        "It seems that your query is still too ambiguous. "
                        "Please try refining your query by including more specific details such as the specific research domain, time period, "
                        "or context you are interested in. "
                        "Examples: 'renewable energy innovations in 2020', 'advances in battery technology for electric vehicles', "
                        "or 'environmental impact of plastic recycling methods'."
                    )
                }
                logger.debug("Maximum clarification rounds reached. Fallback response: %s", fallback)
                return [fallback]

        self.state.clarification_round = 0
        logger.debug("Final results obtained with %d topics.", len(results))
        return results

    def exclude_topics(self, topic_ids: List[str]) -> None:
        """Add topics to the excluded set."""
        self.state.excluded_topic_ids.update(topic_ids)
        logger.debug("Excluded topics updated: %s", self.state.excluded_topic_ids)

    def reset_memory(self) -> None:
        """Reset the agent's memory (both excluded topics and query history)."""
        self.state = AgentState()
        logger.debug("Agent memory reset.")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current agent state for debugging."""
        summary = {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1,
            "clarification_round": self.state.clarification_round
        }
        logger.debug("Agent state summary: %s", summary)
        return summary
