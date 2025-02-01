from typing import Dict, List, Any
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_agent import TopicAgent

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

    def _format_topics(self, topics: List[Dict[str, Any]]) -> str:
        """Format topic results into a readable Markdown message with clickable hyperlinks."""
        if not topics:
            return "No topics found."

        formatted_topics = []
        for topic in topics:
            # Use the topic id to construct a URL.
            # If the id already looks like a URL, use it directly; otherwise, prepend with a base URL.
            topic_id = topic.get("id", "")
            if topic_id.startswith("http"):
                topic_url = topic_id
            else:
                topic_url = f"https://openalex.org/{topic_id}"

            # Format the topic name as a Markdown clickable hyperlink.
            display_name = topic.get("display_name", "Unnamed Topic")
            topic_link = f"**[{display_name}]({topic_url})**"

            # Get the description (or a fallback message).
            description = topic.get("description", "No description available.")

            # Combine the link and description, separating topics by a horizontal rule.
            formatted_topic = f"{topic_link}\n\n{description}\n"
            formatted_topics.append(formatted_topic)

        return "\n---\n".join(formatted_topics)

    def handle_message(self, user_message: str) -> str:
        """
        Main message handler. Searches for topics and formats response.

        Args:
            user_message: The user's input message

        Returns:
            Response string to display to user
        """
        # Search for topics based on user input.
        topics = self.topic_agent.process_query(user_message)

        if not topics:
            return ("I couldn't find any relevant research topics. Could you try rephrasing "
                    "your query or being more specific?")

        return self._format_topics(topics)

    def exclude_current_topics(self, topic_ids: List[str]) -> None:
        """Exclude topics from future searches."""
        self.topic_agent.exclude_topics(topic_ids)

    def reset_conversation(self) -> None:
        """Reset the conversation and agent memory."""
        self.topic_agent.reset_memory()
