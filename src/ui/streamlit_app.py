from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from services.chat_agent import ChatManager
import streamlit.components.v1 as components  # Ensure this import
from streamlit_cytoscapejs import st_cytoscapejs
import json

def load_d3_html(filepath: str) -> str:
    """Load the D3.js HTML file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # This is critical. We need to add in the place to remember what is the
    # final and last input of the user for the generation of JSON
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    # And now for the graph data you've been waiting for days for
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = st.session_state.chat_manager.topic_agent.get_graph_data()

def main():
    st.set_page_config(page_title="Socratic RAG Agent", layout="wide")
    st.title("Socratic RAG Agent")

    # Initialize session state
    initialize_session_state()

    # 1) Add a text input dedicated to autocomplete
    #    Streamlit re-runs your script every time the user types by default.
    partial_input = st.text_input("Type a keyword to auto-complete")

    # 2) Query the autocomplete method
    if partial_input:
        suggestions = st.session_state.chat_manager.autocomplete(partial_input, limit=5)
        if suggestions:
            st.write("**Suggestions:**")
            for s in suggestions:
                st.write(f"- {s}")
        else:
            st.write("No matches found...")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar Controls
    with st.sidebar:
        st.header("Options")
        rag_mode = st.checkbox("Enable RAG Mode?")
        generate_json_button = st.button("Generate JSON Structure?")
        classify_button = st.button("Classify Across Ontologies")
        reset_button = st.button("Reset Conversation")
        # Feedback Form in Sidebar
        st.header("User Feedback")
        with st.form("Feedback Form"):
            st.write("We value your feedback to improve our service.")
            rating = st.slider("Rate the search results", 1, 5, 5)
            comment = st.text_area("Additional comments", height=100)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                user_id = None  # Replace with actual user ID if you have authentication
                st.session_state.chat_manager.handle_feedback(user_id, rating, comment)
                st.success("Thank you for your feedback!")

    # Handle Reset Conversation
    if reset_button:
        st.session_state.messages = []
        st.session_state.chat_manager.reset_conversation()
        st.session_state.last_user_input = ""
        st.experimental_rerun()

    # Normal chat input
    user_input = st.chat_input("What research topics are you interested in? Enter your query to classify:")
    if user_input:
        # Store user input
        st.session_state.last_user_input = user_input
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        # Handle user input
        if rag_mode:
            response = st.session_state.chat_manager.handle_rag_message(user_input)
        else:
            response = st.session_state.chat_manager.handle_message(user_input)
        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to message history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Handle Generate JSON Structure Button
    if generate_json_button:
        if st.session_state.last_user_input:
            structured_data = st.session_state.chat_manager.generate_structured_ontology(st.session_state.last_user_input)
            st.write("**Structured JSON Output:**")
            st.json(structured_data)
        else:
            st.warning("Please enter a query before generating JSON. No query text is available.")

    # Handle Classify Across Ontologies Button
    if classify_button:
        user_topic = st.session_state.last_user_input
        if not user_topic:
            st.warning("No user query to classify. Please enter a query first.")
        else:
            alignment = st.session_state.chat_manager.handle_classification_message(user_topic)
            st.write("**Ontology Alignment Results:**")
            st.json(alignment)

    # Person Search Section
    st.header("Search for a Person")
    person_query = st.text_input("Search for a person by name or affiliation")
    if person_query:
        # search persons
        matched_persons = st.session_state.chat_manager.topic_agent.retrieve_persons(person_query)
        if matched_persons:
            st.write("Found the following persons:")
            for p in matched_persons:
                st.write(f"- {p['name']} ({p['affiliation']})")

            # Let user pick one to see JSON-LD
            selected_name = st.selectbox("Select a person to see JSON-LD", [p["name"] for p in matched_persons])
            if selected_name:
                # get that person's data
                person_data = next(x for x in matched_persons if x["name"] == selected_name)
                jsonld = st.session_state.chat_manager.topic_agent.generate_schema_org_person(person_data)
                st.write("**Schema.org JSON-LD:**")
                st.json(json.loads(jsonld))  # nicely formatted
        else:
            st.write("No persons found for that query.")

    # -------------------------------
    # 6. Gather User Feedback with Forms
    # -------------------------------
    # Note: Feedback form moved to sidebar for better UI.

    # Create Tabs for Different Visualizations
    st.header("Visualizations")
    tabs = st.tabs(["Knowledge Graph", "D3.js Bar Chart"])

    # Knowledge Graph Tab
    with tabs[0]:
        st.subheader("Knowledge Graph")

        graph_data = st.session_state.graph_data

        # Sidebar Graph Filters within the Knowledge Graph Tab
        with st.expander("Graph Filters"):
            node_types = ['Topic', 'Grant', 'Patent', 'Conference', 'Person']
            selected_types = st.multiselect("Select node types to display:", node_types, default=node_types)
            layout_options = st.selectbox("Select Layout:", ["cose", "breadthfirst", "grid", "circle"])

        # Filter nodes and edges based on selected types
        filtered_nodes = [node for node in graph_data['nodes'] if node['data']['type'] in selected_types]
        filtered_edges = [edge for edge in graph_data['edges'] if
                          any(node['data']['id'] == edge['data']['source'] for node in filtered_nodes) and
                          any(node['data']['id'] == edge['data']['target'] for node in filtered_nodes)]

        filtered_graph = filtered_nodes + filtered_edges

        # Define enhanced stylesheet for Cytoscape.js
        stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'color': '#fff',
                    'text-outline-width': 2,
                    'text-outline-color': '#888',
                    'width': 'label',
                    'height': 'label',
                    'padding': '10px',
                    'font-size': '12px',
                    'background-color': '#007BFF',
                    'shape': 'ellipse',
                    'transition-property': 'background-color, shape',
                    'transition-duration': '0.5s'
                }
            },
            {
                'selector': 'node[type = "Grant"]',
                'style': {
                    'background-color': '#28a745',
                    'shape': 'diamond'
                }
            },
            {
                'selector': 'node[type = "Patent"]',
                'style': {
                    'background-color': '#ffc107',
                    'shape': 'rectangle'
                }
            },
            {
                'selector': 'node[type = "Conference"]',
                'style': {
                    'background-color': '#17a2b8',
                    'shape': 'hexagon'
                }
            },
            {
                'selector': 'node[type = "Person"]',
                'style': {
                    'background-color': '#6f42c1',
                    'shape': 'star'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 2,
                    'line-color': '#ccc',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '10px',
                    'text-rotation': 'autorotate',
                    'color': '#555',
                    'text-background-color': '#fff',
                    'text-background-opacity': 1,
                    'text-background-padding': '3px'
                }
            },
            {
                'selector': ':selected',
                'style': {
                    'background-color': '#f00',
                    'line-color': '#f00',
                    'target-arrow-color': '#f00',
                    'source-arrow-color': '#f00',
                    'color': '#fff',
                    'text-outline-color': '#f00'
                }
            }
        ]

        # Render Cytoscape.js Graph with enhanced features
        st_cytoscapejs(
            elements=filtered_graph,
            stylesheet=stylesheet,
            # layout={'name': layout_options, 'animate': True},
            # style={'width': '100%', 'height': '600px'}
        )

    # D3.js Bar Chart Visualization Tab
    with tabs[1]:
        st.subheader("D3.js Bar Chart")

        # Sample dynamic data from Python
        dynamic_data = [
            {"name": "Topic A", "value": 30},
            {"name": "Topic B", "value": 80},
            {"name": "Topic C", "value": 45},
            {"name": "Topic D", "value": 60},
            {"name": "Topic E", "value": 20},
            {"name": "Topic F", "value": 90},
            {"name": "Topic G", "value": 55},
        ]

        # Path to your D3.js HTML file relative to this script
        current_dir = Path(__file__).parent
        d3_html_path = current_dir.parent.parent / "assets" / "d3" / "bar_chart_dynamic.html"

        if d3_html_path.exists():
            d3_html = load_d3_html(str(d3_html_path))
            # Inject dynamic data into the HTML
            d3_html = d3_html.replace("{{ data }}", json.dumps(dynamic_data))
            components.html(
                d3_html,
                height=550,  # Adjust height as needed for better visibility
                scrolling=True
            )
        else:
            st.warning("D3.js HTML file not found. Please ensure 'bar_chart_dynamic.html' exists in the 'assets/d3/' directory.")

if __name__ == "__main__":
    main()
