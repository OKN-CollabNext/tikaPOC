import sys
import os
import json
import pathlib
import streamlit as st
import streamlit.components.v1 as components

# Append parent directory for module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chat_agent import ChatManager
from ui.visualization import create_topic_graph

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
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Reset conversation button in the sidebar
    with st.sidebar:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.experimental_rerun()

    # Visualization section for topic network graph
    with st.expander("Show Topic Network Graph"):
        data_file = "data/openalex_topics_raw.json"
        if os.path.exists(data_file):
            try:
                # Load topics (limit to 50 for demonstration)
                with open(data_file, "r", encoding="utf-8") as f:
                    topics = json.load(f)[:50]

                # Create the topic graph HTML file (this function returns the path to the saved file)
                html_path = create_topic_graph(topics)

                # Read the HTML content from the generated file
                html_content = pathlib.Path(html_path).read_text(encoding="utf-8")

                # Embed the HTML content inline using Streamlit components
                components.html(html_content, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Failed to render topic graph: {str(e)}")
        else:
            st.error(f"Data file not found: {data_file}. Please run the topic processing script to fetch data.")

if __name__ == "__main__":
    main()
