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
        """Format topic results into a readable message. Still using the deployment name from KeyVault as well as falling back to simpler formatting if API fails, I think it would be wise to add more failsafes. """
        try:
            """ Here is the existing call for GPT based upon the formatting... """
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful research assistant. Format the given topics into a "
                            "natural, readable response. For each topic, include its name and description. "
                            "For the name, please turn it into a hyperlink: `(?topic_id=TOPIC_ID)`. "
                            "If the topics seem off-target, suggest how the user might refine their search."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Format these topics into a response: {topics}"
                    }
                ],
                temperature = 0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI A.P.I. error: {str(e)}")
            """ And here's the legendary..fallback formatting if GPT fails.
            We'll turn the display_name into a clickable link using a query param """
            fallback_text = []
            for i, topic in enumerate(topics):
                topic_id = topic["id"]
                display_name = topic["display_name"]
                description = topic["description"]
                """ Here is the structure of the link: "[DisplayName](?topic_id=xxxxxxxx)" """
                link_text = f"[{display_name}](?topic_id={topic_id})"
                fallback_text.append(
                    f"**Topic {i+1}:** {link_text}\n{description}\n"
                )
            return "\n".join(fallback_text)
            """ Hopefully these are enough instructions to turn the display name into a hyperlink with the whole param (?topic_id=TOPIC_ID) thing, so even if GPT does succeed, we still get clickable links from the A.I.. If GPT fails, our fallback does it too..that covers both paths. """

    def handle_message(self, user_message: str, load_more_button_pressed: bool = False) -> str:
        """
        Main message handler. Searches for topics and formats response.

        Args:
            user_message: The user's input message

        Returns:
            Response string to display to user
        """
        # Search for topics
        if load_more_button_pressed:
            """ So that's within the handle_message in our Streamlit function's
            ideal setting.. """
            topics = self.topic_agent.process_query(user_message, next_page=True)
            if not topics:
                return "I couldn't find relevant topics..."
        else:
            topics = self.topic_agent.process_query(user_message, next_page=False)
            if not topics:
                return "I couldn't find relevant topics..."

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
