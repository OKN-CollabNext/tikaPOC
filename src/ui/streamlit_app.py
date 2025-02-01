import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from services.chat_agent import ChatManager

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "orcid_token" not in st.session_state:
        st.session_state.orcid_token = None

def main():
    st.title("Socratic RAG Agent")

    initialize_session_state()

    # Simple ORCID authentication input (stub)
    if not st.session_state.orcid_token:
        token = st.text_input("Enter your ORCID token for login", type="password")
        if st.button("Login"):
            st.session_state.chat_manager.authenticate_researcher(token)
            st.session_state.orcid_token = token
            st.success("Logged in successfully. Please refresh the page if necessary.")
            # Removed st.experimental_rerun() due to version constraints.
            return  # Stop further execution until user refreshes

    # Display chat messages
    st.subheader("Conversation")
    for message in st.session_state.messages:
        # Using simple containers since st.chat_message might not be available in your version
        if message["role"] == "user":
            st.markdown(f"**User:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

    # Chat input with optional location (for geospatial recommendations)
    prompt = st.text_input("What research topics are you interested in?", key="prompt")
    if prompt:
        # Append the user message and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"**User:** {prompt}")

        # Optionally, you could ask for location via additional UI elements (here left as None)
        response = st.session_state.chat_manager.handle_message(prompt, location=None)

        # Append and display the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"**Assistant:** {response}")

    # Sidebar for conversation reset
    with st.sidebar:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.info("Conversation has been reset. Please refresh the page if necessary.")
            # Removed st.rerun() due to version constraints.

if __name__ == "__main__":
    main()
