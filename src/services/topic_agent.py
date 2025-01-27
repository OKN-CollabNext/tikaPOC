import os
import json
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque
from collections import deque
from openai import AzureOpenAI
from .topic_search import TopicSearcher
from .topic_search import get_db_connection
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .ieee_search import IEEESearcher
""" After importing the new IEEE searcher we should also import the new ACM searcher... """
from .acm_search import ACMSearcher
# Then we should import after the IEEE and ACM, the New AMS searcher import
from .ams_search import AMSSearcher
from azure.core.exceptions import ResourceNotFoundError

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
        """ First, and I don't wanna clutter up the comments but try to retrieve the IEEE-API-KEY from the Key Vault.
        """
        # -------------------------------
        # 1) IEEE Key and Searcher Setup
        # -------------------------------
        try:
            """ And here is what we grab, the IEEE API Key from the Key Vault, and or the .env file. """
            self.ieee_api_key = secret_client.get_secret("IEEE-API-KEY").value
            """ And thus we need to instantiate the searcher for IEEE """
            self.ieee_searcher = IEEESearcher(api_key=self.ieee_api_key)
        except ResourceNotFoundError:
            print("Warning: No IEEE-API-KEY found in the Key Vault..which we have moved to..thus the IEEE search will be disabled.")
            self.ieee_api_key = None
            self.ieee_searcher = None
        # -----------------------------
        # 2) ACM Key and Searcher Setup
        # -----------------------------
        try:
            self.acm_api_key = secret_client.get_secret("ACM-API-KEY").value
            self.acm_searcher = ACMSearcher(api_key = self.acm_api_key)
        except ResourceNotFoundError:
            print("Warning: No ACM-API-KEY found in the Key Vault. ACM search disabled.")
            self.acm_api_key = None
            self.acm_searcher = None
        # 3) that's AMS
        try:
            self.ams_api_key = secret_client.get_secret("AMS-API-KEY").value
            self.ams_searcher = AMSSearcher(api_key=self.ams_api_key)
        except ResourceNotFoundError:
            print("Here's a Warning: No AMS-API-KEY found in the Key Vault..AMS Search has been disabled.")
            self.ams_api_key = None
            self.ams_searcher = None
        # ----------------------------
        # 4) AzureOpenAI Setup
        # ----------------------------
        self.client = AzureOpenAI(
            azure_endpoint = secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value

    def classify_topic_across_ontologies(self, user_topic: str) -> Dict[str, str]:
        """
        Returns the mapping of the user_topic to the categories that we recognize..in multiple ontologies (these are ACM, IEEE, AMS. And so on and so forth.).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You must respond with valid JSON **only**, using no extra text. "
                    "Your JSON must have exactly these keys: 'ACM', 'IEEE', 'AMS'. "
                    "Do not include any other keys or text."
                )
            },
            {
                "role": "user",
                "content": f"User topic: {user_topic}"
            }
        ]
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0,
            top_p=1.0,
            max_tokens=200
        )
        raw_text = response.choices[0].message.content.strip()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            print("LLM returned invalid JSON:", raw_text)
            data = {"ACM": None, "IEEE": None, "AMS": None}  # fallback
        return data

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
            # Fallback to using the latest query if rewriting fails
            return query_history[-1]

    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.

        Args:
            user_input: The user's query string

        Returns:
            List of topic dictionaries containing id, display_name, and description
        """
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

        # Search with excluded topics
        return self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids
        )

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
            "has_context": len(self.state.recent_queries) > 1
        }

    def retrieve_context(self, query: str) -> List[str]:
        """
        Here we are ready to retrieve the passages that are relevant, and or the concepts or ontology entries from a multiplicity of sources (OpenAlex, IEEE, ACM, and so on and so forth) based upon the query itself. And the first thing 1) is that we do, we do the Local Results. Which is why we have to remember to retrieve the relevant text passages from the local DataBase plus the external IEEE / ACM. That is how we search the local data-base for matching key-words or topics! Furthermore we have 2) now have to retrieve the IEEE results..if that's available that is..because I don't know and this is my first idea. Further-more, another thing that we need to do is possibly call in the external APIs that we need whether it's IEEE or ACM or additional indexing for the purpose of, for example, if we want to have a function that allows us to query IEEE's database..we get the following...ieee_results = self.ieee_searcher.search(query).
        Further-more we have 3) which means we have the results of ACM if that's even available. What we do is we add in all the lines for searching whether it's by ACM or AMS and then we combine the contexts, we have to combine filter and or rank the results that we get. We combine them and get the local description. And then we're going to want to return the text passages or the structured data for the generator that we have. Here are the IEEE abstracts...Thus we have to add the abstracts external (or any fields relevant)..and we could additionally implement some additional filtering or ranking logic here...last but not least we have the ACM abstracts. And of course if we had a genuine ACM Digital Library API Key, then we would store that in the same Key Vault, the Azure Key Vault. By setting the --vault-name to "tikasecrets" and the --name to "ACM-API-KEY" and the --value to "YOUR_REAL_ACM_API_KEY". Other-wise, I've built in this warning that skips the retrieval of the ACM and the second API (IEEE) for now for that matter.
        So that's the AMS. Of course we're going to have to add in all these AMS IEEE and ACM keys to the Key Vault.
        """
        # 1) First thing is the local topics.
        local_results = self.searcher.search_topics(query)
        # 2) Second thing is the local grants.
        grants_results = self.searcher.search_grants(query)
        # 3) Third thing is the local patents.
        patents_results = self.searcher.search_patents(query)
        # 4) Fourth thing is the local conferences.
        conferences_results = self.searcher.search_conferences(query)
        # 5) External IEEE is the fifth thing.
        ieee_results = []
        if self.ieee_searcher:
            ieee_results = self.ieee_searcher.search(query)
        # 6) External ACM is the sixth thing.
        acm_results = []
        if self.acm_searcher:
            acm_results = self.acm_searcher.search(query)
        # 7) External AMS is the seventh thing.
        ams_results = []
        if self.ams_searcher:
            ams_results = self.ams_searcher.search(query)
        # Combine them seven things...
        combined_contexts = []
        # Add the descriptions of the local topics.
        for topic in local_results:
            combined_contexts.append(topic.get("description", ""))
        # Add the abstracts of the local grants.
        for g in grants_results:
            combined_contexts.append(g.get("abstract", ""))
        # Add the abstracts of the local patents.
        for p in patents_results:
            combined_contexts.append(p.get("abstract", ""))
        # Add the descriptions of the local conferences.
        for c in conferences_results:
            combined_contexts.append(c.get("description", ""))
        # Add the IEEE
        for item in ieee_results:
            combined_contexts.append(item.get("abstract", ""))
        # Add the ACM
        for item in acm_results:
            combined_contexts.append(item.get("abstract", ""))
        # Add the AMS
        for item in ams_results:
            combined_contexts.append(item.get("abstract", ""))
        return combined_contexts

    def autocomplete(self, partial_input: str, limit: int = 5) -> List[str]:
        """
        The simplest approach is to simply make a query on the table `keywords` which allows us to match things partially.
        This returns some suggestions up to the `limit` in order to make sure that they do, begin with `partial_input`.
        """
        if not partial_input:
            return []

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT keyword
                    FROM keywords
                    WHERE keyword LIKE %s
                    LIMIT %s
                """, (partial_input + '%', limit))
                rows = cur.fetchall()

        return [row[0] for row in rows]

    def generate_schema_org_person(self, person_data: Dict[str, Any]) -> str:
        """
        Return JSON-LD conforming to schema.org Person specification.
        person_data is typically something like:
          {
            "name": "Jane Researcher",
            "affiliation": "Example University",
            "profile_url": "https://example.edu/~jane"
          }
        """
        person_schema = {
            "@context": "https://schema.org",
            "@type": "Person",
            "name": person_data.get("name"),
            "affiliation": person_data.get("affiliation"),
            "url": person_data.get("profile_url"),
            # Add any other schema.org fields you'd like,
            # e.g. 'email', 'jobTitle', 'memberOf', etc.
        }
        return json.dumps(person_schema, indent=2)

    def generate_schema_org_organization(self, org_data: Dict[str, Any]) -> str:
        """
        Return JSON-LD conforming to schema.org Organization (or CollegeOrUniversity).
        """
        org_type = org_data.get("institution_type", "Organization")
        # Could be "CollegeOrUniversity", "Company", etc.

        org_schema = {
            "@context": "https://schema.org",
            "@type": org_type,
            "name": org_data.get("name"),
            "url": org_data.get("url"),
            # any other relevant fields
        }
        return json.dumps(org_schema, indent=2)

    def retrieve_persons(self, query: str) -> List[Dict[str, Any]]:
        """
        Example method that calls search_persons to retrieve relevant people.
        """
        return self.searcher.search_persons(query)
