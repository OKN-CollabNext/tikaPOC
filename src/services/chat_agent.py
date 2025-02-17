from typing import Dict, List, Any
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_agent import TopicAgent
from pydantic import BaseModel
import logging

logging.basicConfig(filename="conversation.log", level=logging.INFO, format="%(asctime)s %(message)s")

vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

class MessageClassification(BaseModel):
    is_off_topic: bool
    redirect_message: str | None

class ChatManager:
    def __init__(self) -> None:
        """Initialize the chat manager with the OpenAI client and TopicAgent."""
        self.client = AzureOpenAI(
            azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value
        print(f"Using OpenAI deployment: {self.deployment}")
        self.topic_agent = TopicAgent()

    def _format_topics(self, topics: List[Dict[str, Any]], ambiguous: bool = False) -> str:
        """
        Format topics into a Socratic dialogue response including:
         - Provenance metadata, persistent IDs, and a breakdown of ranking factors.
         - A short KOS summary for each topic.
        """
        note = "It appears your query might have multiple interpretations. " if ambiguous else ""
        breakdown_lines = []
        for t in topics:
            kos_summary = ""
            if "Domain_Specific_KOS" in t.get("source", ""):
                domain = t["source"].split("(")[-1].rstrip(")")
                kos_summary = f"(This topic is from the {domain} domain.)"
            elif t.get("source") == "Broad_KOS":
                kos_summary = "(This topic is from a broad multi-field KOS.)"
            line = (
                f"Topic: {t.get('display_name', 'N/A')} (PID: {t.get('pid', 'N/A')}) | "
                f"Direct: {round(t.get('vector_similarity', 0), 2)} | "
                f"Ancestor Bonus: {round(t.get('bonus', 0), 2)} | "
                f"Temporal Boost: {t.get('trace', {}).get('temporal_boost', 0)} | "
                f"Combined Score: {round(t.get('score', 0), 2)} {kos_summary}"
            )
            breakdown_lines.append(line)
        breakdown = "\n".join(breakdown_lines)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Socratic research guide. For each research topic, provide a brief explanation and ask: "
                    "'Does this direction align with your interests?' Include provenance, persistent IDs, and a breakdown of ranking factors."
                )
            },
            {
                "role": "user",
                "content": f"{note}Here are the candidate topics:\n{breakdown}\n\nWhich aspect best aligns with your research interests?"
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error formatting topics: {e}")
            return "I found some research topics. Could you clarify which aspect you’re most interested in?"

    def _classify_message(self, user_message: str) -> MessageClassification:
        """
        Classify if the incoming message is off-topic.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a message classifier for an academic research topic search system. "
                    "Decide whether the message is suitable for academic research queries. "
                    "If not, provide a redirect message inviting the user to specify a research subject."
                )
            },
            {"role": "user", "content": user_message}
        ]
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=messages,
                response_format=MessageClassification,
                temperature=0.1
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Classification error: {e}")
            return MessageClassification(
                is_off_topic=True,
                redirect_message="Let's focus on academic research topics. What specific subject interests you?"
            )

    def handle_message(self, user_message: str) -> str:
        """
        Main message handler:
         - Logs conversation,
         - Classifies off-topic messages,
         - Passes query to TopicAgent,
         - Handles ambiguous results with a clarifying question,
         - Returns a formatted response with a transparent process trace.
        """
        logging.info(f"User: {user_message}")
        classification = self._classify_message(user_message)
        if classification.is_off_topic:
            response_text = classification.redirect_message or "Let's focus on research topics. What subject would you like to explore?"
            logging.info(f"Assistant: {response_text}")
            return response_text

        topics = self.topic_agent.process_query(user_message)
        if not topics:
            response_text = "I couldn't find any relevant research topics. Could you try rephrasing your query?"
            logging.info(f"Assistant: {response_text}")
            return response_text

        ambiguous = False
        if len(topics) >= 2:
            diff = topics[0].get("score", 0) - topics[1].get("score", 0)
            if diff < 0.1:
                ambiguous = True

        if ambiguous:
            clarifying = self.topic_agent._ask_clarifying_question(topics)
            formatted = self._format_topics(topics, ambiguous=True)
            response_text = f"{formatted}\n\n{clarifying}"
        else:
            response_text = self._format_topics(topics, ambiguous=False)
        logging.info(f"Assistant: {response_text}")
        return response_text

    def exclude_current_topics(self, topic_ids: List[str]) -> None:
        """Exclude topics from future searches."""
        self.topic_agent.exclude_topics(topic_ids)

    def reset_conversation(self) -> None:
        """Reset conversation and agent memory."""
        self.topic_agent.reset_memory()
