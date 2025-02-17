import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from services.chat_agent import ChatManager
import graphviz

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {}

def visualize_graph(topics):
    """
    Generate a simple knowledge graph visualization using Graphviz.
    Nodes show topic display names, persistent IDs, and simulated links to related entities.
    """
    dot = graphviz.Digraph(comment='Research Knowledge Graph')
    dot.node("ROOT", "Research Knowledge Graph")
    for topic in topics:
        label = f"{topic.get('display_name')}\n(PID: {topic.get('pid')})"
        label += f"\nResearcher: Dr. Example (ORCID: 0000-0002-1234-5678)"
        label += f"\nOrg: Example University (ROR: 03yrm5c26)"
        dot.node(topic.get("id"), label)
        dot.edge("ROOT", topic.get("id"))
    return dot

def main():
    st.title("Socratic RAG Agent – Clarify Your Research Direction")
    st.markdown(
        "Welcome! This agent uses a Socratic, multi-round dialogue to help you refine your research topic. "
        "Through iterative questions and clarifications, it makes you think about what you really mean with your precise research intent. "
        "Think of it as a Rosetta Stone for the Socratic spirit--research data across platforms."
    )

    focus = st.sidebar.checkbox("Prioritize emerging/HBCU researchers", value=True)
    if focus:
        st.sidebar.markdown("Underrepresented researchers will be prioritized.")

    if st.sidebar.button("Browse Ontology"):
        ontology_tree = {"Environmental Science": {"Subtopics": ["Waste Management", "Climate Change"]},
                         "Computer Science": {"Subtopics": ["Artificial Intelligence", "Software Engineering"]}}
        st.sidebar.json(ontology_tree)

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

    if st.sidebar.button("Show Knowledge Graph"):
        if st.session_state.messages:
            last_user_query = st.session_state.messages[-1]["content"]
            topics = st.session_state.chat_manager.topic_agent.process_query(last_user_query)
            dot = visualize_graph(topics)
            st.graphviz_chart(dot)
        else:
            st.sidebar.write("No topics available yet.")

    with st.sidebar:
        st.markdown("### Conversation Controls")
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
