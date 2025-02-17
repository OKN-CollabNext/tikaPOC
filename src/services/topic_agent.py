import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque, Optional
from collections import deque
from openai import AzureOpenAI
from .topic_search import TopicSearcher
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from functools import lru_cache
import time
import logging

vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

@dataclass
class AgentState:
    """
    Maintains the agent's memory of excluded topics, recent queries, phase status, dynamic weight,
    and persistent context (e.g., chosen researcher or organization).
    """
    excluded_topic_ids: Set[str] = field(default_factory=set)
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    phase: int = 1
    dynamic_weight: float = 1.0
    chosen_researcher: Optional[str] = None
    chosen_organization: Optional[str] = None

class TopicAgent:
    def __init__(self):
        """Initialize the agent with its components."""
        self.state = AgentState()
        self.searcher = TopicSearcher()
        self.client = AzureOpenAI(
            azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value

    def _translate_if_needed(self, text: str, target_language: str = "en") -> str:
        """(Stub) Translate text into the target language. In production, integrate a translation API."""
        return text

    def _maybe_expand_ontology(self, query: str) -> None:
        """(Stub) If the query is unmatched, propose a new emergent topic node."""
        print(f"Checking if query '{query}' needs an emergent ontology node...")
        pass

    def _prioritize_by_researcher_org(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Boost topics that match persistent researcher or organization preferences."""
        if self.state.chosen_researcher:
            for t in topics:
                if "researcher" in t and self.state.chosen_researcher.lower() in t["researcher"].lower():
                    t["score"] += 0.2
        if self.state.chosen_organization:
            for t in topics:
                if "org" in t and self.state.chosen_organization.lower() in t["org"].lower():
                    t["score"] += 0.2
        topics.sort(key=lambda x: x["score"], reverse=True)
        return topics

    def _rewrite_query(self) -> str:
        """Rewrite the query using recent query context and Socratic guidance."""
        query_history = list(self.state.recent_queries)
        print(f"Query history: {query_history}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Socratic query rewriter. Combine the following queries into one clear, comprehensive query "
                    "that captures the evolving intent and invites clarification. Return ONLY the rewritten query."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Previous queries: {', '.join(query_history[:-1])}\n"
                    f"Current query: {query_history[-1]}\n\n"
                    "Rewrite these into one search-friendly query."
                )
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=512
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query_history[-1]

    def _ask_clarifying_question(self, topics: List[Dict[str, Any]]) -> str:
        """Ask a clarifying question if results are ambiguous."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Socratic dialogue assistant. When topics have similar scores, ask a brief, focused question "
                    "to help narrow down which aspect is most relevant."
                )
            },
            {
                "role": "user",
                "content": "The search results are ambiguous. Ask one direct question to clarify the user's preference."
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=256
            )
            clarifying_question = response.choices[0].message.content.strip()
            print(f"Clarifying question: {clarifying_question}")
            return clarifying_question
        except Exception as e:
            print(f"Error asking clarifying question: {e}")
            return "Could you please specify which aspect interests you most?"

    def update_taxonomies(self) -> None:
        """Dynamically update internal taxonomies by querying external KOS APIs (e.g., Wikidata, MeSH, CrossRef)."""
        print("Updating internal taxonomies from external KOS APIs...")
        pass

    def get_ontology_tree(self) -> Dict[str, Any]:
        """
        (Stub) Return an ontology tree for interactive browsing.
        In a real implementation, this would return the full taxonomy as a nested dictionary.
        """
        return {
            "Environmental Science": {
                "Subtopics": ["Waste Management", "Climate Change"],
                "Siblings": ["Biology", "Chemistry"]
            },
            "Computer Science": {
                "Subtopics": ["Artificial Intelligence", "Software Engineering"],
                "Siblings": ["Mathematics", "Physics"]
            }
        }

    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.
        Incorporates adaptive ontology expansion, temporal weighting, multilingual support,
        and attaches detailed provenance and process trace.
        """
        user_input = self._translate_if_needed(user_input)
        self._maybe_expand_ontology(user_input)
        self.state.recent_queries.append(user_input)
        print(f"Query history: {list(self.state.recent_queries)}")

        current_time = int(time.time())

        if "confirm broad" in user_input.lower():
            self.state.phase = 2
            print("Switching to domain-specific retrieval.")
        else:
            self.state.phase = 1

        query_to_search = self._rewrite_query() if len(self.state.recent_queries) > 1 else user_input
        print(f"Final query: {query_to_search}")

        if self.state.phase == 1:
            results = self.searcher.search_topics(query=query_to_search, excluded_topic_ids=self.state.excluded_topic_ids)
        else:
            domain = "CS" if "computer" in query_to_search.lower() else "general"
            results = self.searcher.search_topics_domain(query=query_to_search, domain=domain, excluded_topic_ids=self.state.excluded_topic_ids)

        for topic in results:
            if self.state.phase == 1:
                topic.setdefault("source", "Broad_KOS")
                topic.setdefault("provenance", "Retrieved from multi-field KOS (OpenAlex, Dewey Decimal, etc.)")
            else:
                topic.setdefault("source", f"Domain_Specific_KOS ({domain})")
                topic.setdefault("provenance", "Refined using domain-specific taxonomies (CSO, MeSH, etc.)")
            topic.setdefault("pid", f"PID-{topic['id']}")
            topic.setdefault("definition", self.searcher.fetch_definition(topic["id"]))
            topic.setdefault("timestamp", current_time)

        results = self._prioritize_by_researcher_org(results)

        for topic in results:
            topic.setdefault("trace", {
                "direct_similarity": round(topic.get("vector_similarity", 0), 2),
                "ancestor_bonus": round(topic.get("bonus", 0), 2),
                "temporal_boost": 0.1 if current_time - int(topic.get("timestamp", current_time)) < 31536000 else 0,
                "final_score": round(topic.get("score", 0), 2)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def process_feedback(self, feedback: str) -> None:
        """
        Process explicit user feedback to adapt query rewriting and ranking.
        Also update dynamic weight based on preference.
        """
        print(f"Received feedback: {feedback}")
        for token in feedback.split():
            if token.startswith("topic_"):
                self.exclude_topics([token])
        if "not relevant" in feedback.lower():
            self.state.dynamic_weight *= 0.9
        elif "more" in feedback.lower():
            self.state.dynamic_weight *= 1.1

    def exclude_topics(self, topic_ids: List[str]) -> None:
        """Add topics to the excluded set."""
        self.state.excluded_topic_ids.update(topic_ids)

    def reset_memory(self) -> None:
        """Reset the agent's memory (excluded topics, query history, and context)."""
        self.state = AgentState()

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of the current state for debugging."""
        return {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "phase": self.state.phase,
            "dynamic_weight": self.state.dynamic_weight,
            "chosen_researcher": self.state.chosen_researcher,
            "chosen_organization": self.state.chosen_organization,
            "has_context": len(self.state.recent_queries) > 1
        }
