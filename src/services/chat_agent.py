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

    def _format_topics_with_explanations(self, topics: List[Dict[str, Any]]) -> str:
        """Format topic results into a readable message with explanations."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful research assistant. Format the given topics into a
                natural, readable response. For each topic, include its name, description, and a brief explanation of why it was suggested."""
            },
            {
                "role": "user",
                "content": f"Format these topics into a response: {topics}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,  # Use the deployment name from Key Vault
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error in formatting topics: {str(e)}")
            # Fallback to simple formatting with explanations
            formatted_topics = "\n".join([
                f"**Topic {i+1}: {topic['display_name']}**\n{topic['description']}\n*Suggested based on your interest in related keywords.*"
                for i, topic in enumerate(topics)
            ])
            return formatted_topics

    def handle_message(self, user_message: str) -> str:
        """
        Main message handler. Searches for topics and formats response.

        Args:
            user_message: The user's input message

        Returns:
            Response string to display to user
        """
        if self.topic_agent.state.awaiting_clarification:
            # Handle the user's clarification response
            topics = self.topic_agent.handle_clarification_response(user_message)

            if not topics:
                return ("I still couldn't find any relevant research topics based on your clarification. "
                        "Could you please provide more details or refine your query further?")

            return self._format_topics_with_explanations(topics)

        # Process the user's initial query
        topics = self.topic_agent.process_query(user_message)

        if not topics:
            return ("I couldn't find any relevant research topics. Could you try rephrasing "
                   "your query or being more specific?")

        # Check if multiple topics were returned, suggesting need for clarification
        if len(topics) > 1:
            # Generate a clarifying question
            clarifying_question = self._generate_clarifying_question(topics)
            return clarifying_question

        # Format and return topics with explanations
        return self._format_topics_with_explanations(topics)

    def _generate_clarifying_question(self, topics: List[Dict[str, Any]]) -> str:
        try:
            topic_names = [topic['display_name'] for topic in topics]
            question = ("I found multiple topics related to your query. Could you please specify which aspect you're interested in?\n"
                        f"Here are some of the topics I found: {', '.join(topic_names[:5])}.\n"
                        "Please provide more details or select one of these topics.")
            return question
        except Exception as e:
            print(f"Error generating clarifying question: {e}")
            return "Error processing your request."

    def exclude_current_topics(self, topic_ids: List[str]) -> None:
        """Exclude topics from future searches."""
        self.topic_agent.exclude_topics(topic_ids)

    def reset_conversation(self) -> None:
        """Reset the conversation and agent memory."""
        self.topic_agent.reset_memory()
