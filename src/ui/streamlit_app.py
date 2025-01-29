import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from services.chat_agent import ChatManager
import json

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Socratic RAG Agent")

    # Initialize session state
    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What research topics are you interested in?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a reset button in the sidebar
    with st.sidebar:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.experimental_rerun()

    # -------------------------------
    # New Functionalities Integration
    # -------------------------------
    st.header("Advanced Functionalities")

    # Classify Across Ontologies
    st.subheader("Classify Topic Across Ontologies")
    user_topic = st.text_input("Enter a topic to classify across ACM, IEEE, and AMS:")
    if st.button("Classify"):
        if user_topic:
            classification = st.session_state.chat_manager.classify_topic_across_ontologies(user_topic)
            st.json(classification)
        else:
            st.warning("Please enter a topic to classify.")

    # Generate JSON Structure
    st.subheader("Generate Structured JSON Ontology")
    json_input = st.text_input("Enter input to generate structured JSON ontology:")
    if st.button("Generate JSON Structure"):
        if json_input:
            structured_json = st.session_state.chat_manager.generate_structured_ontology(json_input)
            st.json(structured_json)
        else:
            st.warning("Please enter input to generate JSON structure.")

if __name__ == "__main__":
    main()
