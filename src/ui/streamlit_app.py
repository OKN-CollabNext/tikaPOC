# File: /Users/deangladish/tikaPOC/src/ui/streamlit_app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import streamlit as st
from services.chat_agent import ChatManager
from typing import List

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "autocomplete_active" not in st.session_state:
        st.session_state.autocomplete_active = False
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []
    if "selected_keyword" not in st.session_state:
        st.session_state.selected_keyword = ""

def main():
    st.set_page_config(page_title="Socratic RAG Agent", layout="wide")
    st.title("Socratic RAG Agent")

    # Initialize session state
    initialize_session_state()

    # Sidebar with reset button
    with st.sidebar:
        st.header("Options")
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.session_state.autocomplete_active = False
            st.session_state.suggestions = []
            st.session_state.selected_keyword = ""
            st.experimental_rerun()

    # Display chat messages
    st.header("Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Autocomplete Input for Research Topics
    st.subheader("Search for Research Topics")
    with st.form("autocomplete_form", clear_on_submit=False):
        partial_input = st.text_input("Enter a research topic keyword:")
        submitted = st.form_submit_button("Get Suggestions")

        if submitted and partial_input:
            try:
                # Fetch suggestions
                suggestions = st.session_state.chat_manager.autocomplete(partial_input, limit=5)
                st.session_state.suggestions = suggestions
                st.session_state.autocomplete_active = True
            except Exception as e:
                st.error(f"Autocomplete failed: {e}")
                st.session_state.chat_manager.logger.error(f"Autocomplete error: {e}")

    # Display Suggestions if active
    if st.session_state.autocomplete_active and st.session_state.suggestions:
        st.subheader("Suggestions")
        selected_keyword = st.selectbox(
            "Select a keyword",
            options=st.session_state.suggestions,
            index=0,
            key="selectbox_suggestions"
        )

        if st.button("Use Selected Keyword"):
            if selected_keyword:
                prompt = selected_keyword
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get bot response
                try:
                    response = st.session_state.chat_manager.handle_message(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Failed to process your input: {e}")
                    st.session_state.chat_manager.logger.error(f"Error processing user input: {e}")

                # Reset autocomplete state
                st.session_state.autocomplete_active = False
                st.session_state.suggestions = []
                st.session_state.selected_keyword = ""

    # Chat Input for General Queries
    st.subheader("Ask a Question")
    if prompt := st.chat_input("What research topics are you interested in?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        try:
            response = st.session_state.chat_manager.handle_message(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Failed to process your input: {e}")
            st.session_state.chat_manager.logger.error(f"Error processing user input: {e}")

if __name__ == "__main__":
    main()
