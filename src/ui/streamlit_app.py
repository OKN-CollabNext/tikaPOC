import sys
import os
import re
import pandas as pd
import streamlit as st

# Add parent directory to sys.path so we can import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.chat_agent import ChatManager

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []

def visualize_query_execution_times(log_file_path: str):
    """
    Reads a log file, extracts query execution times, and displays a line chart in Streamlit.

    The function looks for lines containing a pattern like:
        Execution Time: 2314.467 ms
    """
    execution_times = []
    query_numbers = []

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                # Regular expression to extract the execution time in milliseconds
                match = re.search(r'Execution Time:\s*([\d\.]+)\s*ms', line)
                if match:
                    execution_times.append(float(match.group(1)))
                    query_numbers.append(len(query_numbers) + 1)
    except FileNotFoundError:
        st.error(f"Log file not found: {log_file_path}")
        return

    if execution_times:
        # Create a DataFrame for visualization
        df = pd.DataFrame({
            "Query Number": query_numbers,
            "Execution Time (ms)": execution_times
        })

        st.write("## Query Execution Times")
        st.write("This chart shows the execution time (in milliseconds) for each logged query.")

        # Display a line chart with Query Number as the x-axis
        st.line_chart(df.set_index("Query Number"))

        # Also display the raw data in a table for reference
        st.write("### Detailed Data")
        st.dataframe(df)
    else:
        st.info("No query execution times found in the log file.")

def chat_ui():
    """Display the chat user interface."""
    st.title("Socratic RAG Agent - Chat")
    initialize_session_state()

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input from the user
    if prompt := st.chat_input("What research topics are you interested in?"):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display the bot response
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Reset conversation button in the sidebar
    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.chat_manager.reset_conversation()
        st.experimental_rerun()

def dashboard_ui():
    """Display the dashboard UI for query execution time visualization."""
    st.title("Socratic RAG Agent - Dashboard")
    st.header("Query Execution Time Dashboard")

    # Allow the user to upload a log file
    uploaded_file = st.file_uploader("Upload log file (optional)", type=["log", "txt"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        temp_log_path = "temp_log_file.log"
        with open(temp_log_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        log_file = temp_log_path
    else:
        # Use a default log file path if no file is uploaded
        log_file = "app.log"
        if not os.path.exists(log_file):
            st.info("Default log file 'app.log' not found. Please upload a log file.")
            return

    visualize_query_execution_times(log_file)

def main():
    # Sidebar navigation to choose between Chat and Dashboard views
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Select View", ("Chat", "Dashboard"))

    if view == "Chat":
        chat_ui()
    elif view == "Dashboard":
        dashboard_ui()

if __name__ == "__main__":
    main()
