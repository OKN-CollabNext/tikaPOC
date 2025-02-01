import sys
import os
import re
import pandas as pd
import streamlit as st

# Add the project root to sys.path so that modules in services are found.
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
    Reads a log file, extracts query execution times, and displays a line chart.
    Looks for lines like:
        Execution Time: 2314.467 ms
    """
    execution_times = []
    query_numbers = []
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = re.search(r'Execution Time:\s*([-+]?[0-9]*\.?[0-9]+)\s*ms', line)
                if match:
                    execution_times.append(float(match.group(1)))
                    query_numbers.append(len(query_numbers) + 1)
    except FileNotFoundError:
        st.error(f"Log file not found: {log_file_path}")
        return

    if execution_times:
        df = pd.DataFrame({
            "Query Number": query_numbers,
            "Execution Time (ms)": execution_times
        })
        st.write("## Query Execution Times")
        st.line_chart(df.set_index("Query Number"))
        st.write("### Detailed Execution Time Data")
        st.dataframe(df)
    else:
        st.info("No query execution times found in the log file.")

def visualize_similarity_metrics(log_file_path: str):
    """
    Reads the log file and extracts similarity metrics:
      - 'Vector Similarity: <value>'
      - 'Keyword Similarity: <value>'
      - 'Combined Score: <value>'
    Then displays these metrics as line charts.
    """
    vector_vals = []
    keyword_vals = []
    combined_vals = []
    occurrences = []
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                vec_match = re.search(r'Vector Similarity:\s*([-+]?[0-9]*\.?[0-9]+)', line)
                key_match = re.search(r'Keyword Similarity:\s*([-+]?[0-9]*\.?[0-9]+)', line)
                comb_match = re.search(r'Combined Score:\s*([-+]?[0-9]*\.?[0-9]+)', line)
                if vec_match and key_match and comb_match:
                    vector_vals.append(float(vec_match.group(1)))
                    keyword_vals.append(float(key_match.group(1)))
                    combined_vals.append(float(comb_match.group(1)))
                    occurrences.append(len(occurrences) + 1)
    except FileNotFoundError:
        st.error(f"Log file not found: {log_file_path}")
        return

    if vector_vals:
        df = pd.DataFrame({
            "Occurrence": occurrences,
            "Vector Similarity": vector_vals,
            "Keyword Similarity": keyword_vals,
            "Combined Score": combined_vals
        })
        st.write("## Similarity Metrics")
        st.line_chart(df.set_index("Occurrence"))
        st.write("### Detailed Similarity Data")
        st.dataframe(df)
    else:
        st.info("No similarity metrics found in the log file.")

def chat_ui():
    """Display the chat user interface."""
    st.title("Socratic RAG Agent - Chat")
    initialize_session_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What research topics are you interested in?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.chat_manager.reset_conversation()
        st.experimental_rerun()

def dashboard_ui():
    """Display the dashboard UI for metrics visualization."""
    st.title("Socratic RAG Agent - Dashboard")
    st.header("Metrics Dashboard")

    # Allow the user to upload a log file; otherwise, use the default "app.log".
    uploaded_file = st.file_uploader("Upload log file (optional)", type=["log", "txt"])
    if uploaded_file is not None:
        temp_log_path = "temp_log_file.log"
        with open(temp_log_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        log_file = temp_log_path
    else:
        log_file = "app.log"
        if not os.path.exists(log_file):
            st.info("Default log file 'app.log' not found. Please upload a log file.")
            return

    st.subheader("Query Execution Times")
    visualize_query_execution_times(log_file)

    st.subheader("Similarity Metrics")
    visualize_similarity_metrics(log_file)

def main():
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Select View", ("Chat", "Dashboard"))
    if view == "Chat":
        chat_ui()
    elif view == "Dashboard":
        dashboard_ui()

if __name__ == "__main__":
    main()
