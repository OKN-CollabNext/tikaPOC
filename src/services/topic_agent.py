import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque, Optional
from collections import deque
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
# That might slow down our app if it downloads every time..
# so we consider that a check
from nltk.corpus import wordnet # So that we can have the alternate way to expand synonyms...
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
    """Maintains agent's memory of excluded topics and recent queries, and pagination."""
    excluded_topic_ids: Set[str] = field(default_factory=set)
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    current_page: int = 0 #And now we should track the current "page" for an in-finite scroll
    page_size: int = 10 #that's how many results we have per-page.

class TopicAgent:
    def __init__(self):
        """Initialize the agent with necessary components and pagination logic."""
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
                "content": """You are a query rewriting assistant. Given a sequence of user queries,
                rewrite them into a single, comprehensive query that captures the user's evolving intent.
                Focus on creating a search-friendly query that works well with embedding-based search.
                Return ONLY the rewritten query, nothing else."""
            },
            {
                "role": "user",
                "content": f"""Previous queries: {', '.join(query_history[:-1])}
                Current query: {query_history[-1]}

                Rewrite these queries into a single, comprehensive query."""
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
            # Fallback to using the latest user query "if rewriting fails"
            return query_history[-1]

    def _expand_synonyms(self, text: str) -> str:
        """ Well, now we can expand the synonyms using WordNet.
        Hopefully this exemplifies how if I don't want it I don't need to
        call the function. """
        words = text.split()
        expansions = []
        for w in words:
            synsets = wordnet.synsets(w)
            if synsets:
                # Gather some two synonyms to demonstrate..
                synonyms = {
                    lemma.name().replace("_", " ")
                    for syn in synsets[:2] # continues onwards to first two synsets
                    for lemma in syn.lemmas()
                }
                expansions.append(w + " (" + ", ".join(synonyms) + ")")
            else:
                expansions.append(w)
        return " ".join(expansions)

    def process_query(self, user_input: str, next_page: bool = False) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.
        Now we have the option to do some pagination..so that instead of messing
        with line breaks and stuff we can say that if we want to handle "Load More"
        or something like "Next Page," then we can by adding in the parameteric..
        the (e.g. next_page=True) and then increment the current_page.

        Args:
            user_input: The user's query string

        Returns:
            List of topic dictionaries containing id, display_name, and description
        """
        # For example. Let's say you want to do an approach for infinite-scroll:
        # re-set that if the query's new
        if next_page:
            self.state.current_page += 1
        else:
            self.state.current_page = 0

        # Add new query to history
        self.state.recent_queries.append(user_input)

        # Debug print
        print(f"Current query history: {list(self.state.recent_queries)}")
        print(f"Number of queries in history: {len(self.state.recent_queries)}")

        # Determine if we should rewrite
        should_rewrite = len(self.state.recent_queries) > 1
        print(f"Should rewrite: {should_rewrite}")  # Debug print

        # Get search query
        query_to_search = self._rewrite_query() if should_rewrite else user_input
        print(f"Final search query: {query_to_search}")  # Debug print

        # And now we could expand out the synonyms..leave it commented out
        # to avoid "noisy" expansions.
        query_to_search = self._expand_synonyms(query_to_search)

        # Search with excluded topics
        return self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids,
            n_similar_keywords=10,
            n_topics=self.state.page_size,
            #you can also add in some off-set pagination based in search_topics
            # e.g. pass (offset = self.state.current_page * self.state.page_size)
            offset = self.state.current_page * self.state.page_size
            # ANd then do a "LIMIT page_size OFFSET offset" in your SQL
        )

    def exclude_topics(self, topic_ids: List[str]) -> None:
        """Add topics to the excluded set."""
        self.state.excluded_topic_ids.update(topic_ids)

    def reset_memory(self) -> None:
        """Reset the agent's memory (both excluded topics and query history)."""
        self.state = AgentState()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current agent state for debugging."""
        summary = {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1,
            "current_page": self.state.current_page,
            "page_size": self.state.page_size
        }
        return summary
