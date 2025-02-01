from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_agent import TopicAgent
from .orcid_auth import orcid_authenticate  # New: stub for ORCID-based authentication

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

class ChatManager:
    def __init__(self) -> None:
        """Initialize chat manager with OpenAI client and topic agent."""
        self.client = AzureOpenAI(
            azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value
        print(f"Using OpenAI deployment: {self.deployment}")
        self.topic_agent = TopicAgent()
        # New: Store authenticated researcher info
        self.researcher_profile = None

    def authenticate_researcher(self, orcid_token: str) -> None:
        """
        Authenticate the researcher using ORCID.
        In production, this would verify the ORCID token and return profile details.
        """
        self.researcher_profile = orcid_authenticate(orcid_token)
        print(f"Authenticated researcher: {self.researcher_profile}")

    def _format_topics(self, topics: List[Dict[str, Any]]) -> str:
        """Format topic results into a readable message with hyperlink navigation."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. Format the given topics into a natural, readable response. "
                    "For each topic, include its name (as a clickable link) and description. If the topics seem off-target, "
                    "suggest how the user might refine their search."
                )
            },
            {
                "role": "user",
                "content": f"Format these topics into a response: {topics}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            # Fallback to simple formatting with hyperlinks (dummy URLs)
            return "\n".join([
                f"[Topic {i+1}: {topic['display_name']}](https://example.org/topic/{topic['id']})\n{topic['description']}"
                for i, topic in enumerate(topics)
            ])

    def handle_message(self, user_message: str, location: Optional[Dict[str, float]] = None) -> str:
        """
        Main message handler. Searches for topics and formats response.

        Args:
            user_message: The user's input message.
            location: Optional location data for geospatial recommendations.

        Returns:
            Response string to display to user.
        """
        # Search for topics (pass along location info)
        topics = self.topic_agent.process_query(user_message, location=location)

        if not topics:
            return (
                "I couldn't find any relevant research topics. Could you try rephrasing your query or being more specific?"
            )

        return self._format_topics(topics)

    def exclude_current_topics(self, topic_ids: List[str]) -> None:
        """Exclude topics from future searches."""
        self.topic_agent.exclude_topics(topic_ids)

    def reset_conversation(self) -> None:
        """Reset the conversation and agent memory."""
        self.topic_agent.reset_memory()
