from typing import Dict, List, Any
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_agent import TopicAgent
import json

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

    def handle_classification_message(self, user_topic: str) -> Dict[str, str]:
        return self.topic_agent.classify_topic_across_ontologies(user_topic)

    def _format_topics(self, topics: List[Dict[str, Any]]) -> str:
        """Format topic results into a readable message."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful research assistant. Format the given topics into a
                natural, readable response. For each topic, include its name and description.
                If the topics seem off-target, suggest how the user might refine their search."""
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
            print(f"OpenAI API error: {str(e)}")
            # Fallback to simple formatting if API fails
            return "\n".join([
                f"Topic {i+1}: {topic['display_name']}\n{topic['description']}"
                for i, topic in enumerate(topics)
            ])

    def handle_rag_message(self, user_message: str) -> str:
        """
        In this method, I will call specifically the generate_rag_response
        in order to produce an answer using the method of RAG retrieval.
        """
        return self.generate_rag_response(user_message)

    def handle_message(self, user_message: str) -> str:
        """
        Main message handler. Searches for topics and formats response.

        Args:
            user_message: The user's input message

        Returns:
            Response string to display to user
        """
        # Search for topics
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

    # ---------------------------------------------------------
    # NEW RAG-STYLE METHOD
    # ---------------------------------------------------------
    def generate_rag_response(self, query: str) -> str:
        """ Here, we should use the context as we have retrieved it in order to engage in the generation of an answer in the style of R.A.G..  """
        # Such as such the first thing that we are going to want to do is retrieve the relevant context from the agent.
        context_snippets = self.topic_agent.retrieve_context(query)
        # And thus we are going to want to build in the prompt with the relevant context.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant that uses the provided context "
                    "to answer questions. Only use the context if relevant."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    "Context:\n" +
                    "\n\n".join(context_snippets) +
                    "\n\nPlease provide an answer using only the context if possible."
                )
            }
        ]
        # 3. And here is where we send in the request to the OpenAI API.
        response = self.client.chat.completions.create(
            model = self.deployment,
            messages = messages,
            temperature = 0.3,
            max_tokens = 250
        )

        return response.choices[0].message.content.strip()
        """ And here is how we do it. We have to generate the RAG response which means
        that we call in the self.topic_agent.retrieve_context(query) which is what allows us to get the text passages from our local database plus the IEEE/ACM/AMS.
        Then, we can **inject** those passages into the prompt GPT and then return a summarized RAG-style answer. """
    def generate_structured_ontology(self, user_input: str) -> Dict[str, Any]:
        """
        Here is where we create a mapping of the ontology structured in JSON,
        for the user input that we are given. Here we are going to return a dictionary
        in Python with the following fields: these fields are the 'id', the 'name', the 'ontology_category', as well as the 'relations'.
        """
        # And here is what we do first. First, we want to build the user messages
        # as well as the system messages that are going to instruct the LLM.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that outputs structured JSON. "
                    "Always return valid JSON with the keys: ['id', 'name', 'ontology_category', 'relations'] "
                    "and do not include any extra text."
                )
            },
            {
                "role": "user",
                "content": f"Classify the query '{user_input}' according to our structured ontology."
            }
        ]
        # Then, we are going to secondly call the OpenAI with Azure.
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0.0,
            max_tokens=300
        )
        # Furthermore, we're going to want to take the JSON from the response of
        # the LLM and parse the JSON from it.
        try:
            structured_output = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # If the model somehow returns JSON that isn't valid, then we want to
            # gracefully handle it.
            structured_output = {
                "id": None,
                "name": None,
                "ontology_category": None,
                "relations": []
            }

        return structured_output

    def autocomplete(self, partial_input: str, limit: int = 5) -> List[str]:
        """
        Delegate to the method of autocomplete of TopicAgent.
        """
        return self.topic_agent.autocomplete(partial_input, limit)
