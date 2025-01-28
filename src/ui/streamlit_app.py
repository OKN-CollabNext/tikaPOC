# File: /Users/deangladish/tikaPOC/src/ui/streamlit_app.py

from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import logging
import json
from PIL import Image
import streamlit.components.v1 as components
from streamlit_cytoscapejs import st_cytoscapejs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path for module imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from services.chat_agent import ChatManager  # Ensure this path is correct

def add_custom_css():
    """Adds custom CSS to the Streamlit app for styling."""
    st.markdown(
        """
        <style>
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }

        /* Button styling */
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
        }

        /* Header styling */
        h1 {
            color: #1E90FF;
            text-align: center;
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #1E90FF;
        }

        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 140px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -70px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def load_d3_html(filepath: Path) -> str:
    """Loads the D3.js HTML file.

    Args:
        filepath (Path): Path to the D3.js HTML file.

    Returns:
        str: Contents of the HTML file.
    """
    try:
        with filepath.open('r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"D3.js HTML file not found at {filepath}")
        return ""

def initialize_session_state():
    """Initializes session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
        logger.info("Initialized ChatManager in session state.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized messages list in session state.")
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
        logger.info("Initialized last_user_input in session state.")
    if "graph_data" not in st.session_state:
        try:
            st.session_state.graph_data = st.session_state.chat_manager.topic_agent.get_graph_data()
            logger.info("Loaded graph data into session state.")
        except Exception as e:
            st.session_state.graph_data = {'nodes': [], 'edges': []}
            logger.error(f"Failed to load graph data: {e}")

def get_asset_path(relative_path: str) -> Path:
    """Returns the absolute path to a file in the assets directory.

    Args:
        relative_path (str): Relative path within the assets directory.

    Returns:
        Path: Absolute Path object.
    """
    assets_dir = project_root / "assets"
    file_path = assets_dir / relative_path
    return file_path

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Socratic RAG Agent", layout="wide")
    add_custom_css()
    st.title("Socratic RAG Agent")

    # Initialize session state
    initialize_session_state()

    # Sidebar with logo and options
    with st.sidebar:
        # Load and display the logo
        logo_path = get_asset_path("logo.jpg")
        if logo_path.exists():
            try:
                st.image(str(logo_path), use_container_width=True)  # Updated parameter
                logger.info(f"Loaded logo image from {logo_path}")
            except Exception as e:
                st.error(f"Failed to load logo image: {e}")
                logger.error(f"Error loading logo image: {e}")
        else:
            st.error(f"Logo file not found at {logo_path}")
            logger.error(f"Logo file not found at {logo_path}")

        st.header("Options")
        rag_mode = st.checkbox("Enable RAG Mode?")
        generate_json_button = st.button("Generate JSON Structure?")
        classify_button = st.button("Classify Across Ontologies")
        reset_button = st.button("Reset Conversation")

        st.header("User Feedback")
        with st.form("Feedback Form"):
            rating = st.slider("Rate the search results", 1, 5, 5)
            comment = st.text_area("Additional comments", height=100)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                user_id = None  # Replace with actual user ID if available
                try:
                    st.session_state.chat_manager.handle_feedback(user_id, rating, comment)
                    st.success("Thank you for your feedback!")
                    logger.info(f"Received feedback: Rating={rating}, Comment='{comment}'")
                except Exception as e:
                    st.error(f"Failed to submit feedback: {e}")
                    logger.error(f"Error submitting feedback: {e}")

    # Main content layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Keyword Autocomplete")
        partial_input = st.text_input("Type a keyword to auto-complete")
        if partial_input:
            try:
                suggestions = st.session_state.chat_manager.autocomplete(partial_input, limit=5)
                if suggestions:
                    st.markdown("**Suggestions:**")
                    for s in suggestions:
                        st.markdown(f"- {s}")
                else:
                    st.markdown("No matches found...")
            except Exception as e:
                st.error(f"Autocomplete failed: {e}")
                logger.error(f"Autocomplete error: {e}")

    with col2:
        # Load and display the side image
        side_image_path = get_asset_path("side_image.png")
        if side_image_path.exists():
            try:
                st.image(str(side_image_path), use_container_width=True)  # Updated parameter
                logger.info(f"Loaded side image from {side_image_path}")
            except Exception as e:
                st.error(f"Failed to load side image: {e}")
                logger.error(f"Error loading side image: {e}")
        else:
            st.error(f"Side image file not found at {side_image_path}")
            logger.error(f"Side image file not found at {side_image_path}")

    # Chat Messages
    st.header("Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar Controls (Reset handled separately)
    if reset_button:
        st.session_state.messages = []
        try:
            st.session_state.chat_manager.reset_conversation()
            st.session_state.last_user_input = ""
            st.experimental_rerun()
            logger.info("Conversation reset by user.")
        except Exception as e:
            st.error(f"Failed to reset conversation: {e}")
            logger.error(f"Error resetting conversation: {e}")

    # Chat Input
    user_input = st.chat_input("What research topics are you interested in? Enter your query to classify:")
    if user_input:
        st.session_state.last_user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        try:
            if rag_mode:
                response = st.session_state.chat_manager.handle_rag_message(user_input)
                logger.info("Handled RAG message.")
            else:
                response = st.session_state.chat_manager.handle_message(user_input)
                logger.info("Handled standard message.")
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Failed to process your input: {e}")
            logger.error(f"Error processing user input: {e}")

    # Generate JSON Button
    if generate_json_button:
        if st.session_state.last_user_input:
            try:
                structured_data = st.session_state.chat_manager.generate_structured_ontology(st.session_state.last_user_input)
                st.markdown("### **Structured JSON Output:**")
                st.json(structured_data)
                logger.info("Generated structured JSON output.")
            except Exception as e:
                st.error(f"Failed to generate JSON structure: {e}")
                logger.error(f"Error generating JSON structure: {e}")
        else:
            st.warning("Please enter a query before generating JSON.")

    # Classify Across Ontologies Button
    if classify_button:
        user_topic = st.session_state.last_user_input
        if not user_topic:
            st.warning("No user query to classify. Please enter a query first.")
        else:
            try:
                alignment = st.session_state.chat_manager.handle_classification_message(user_topic)
                st.markdown("### **Ontology Alignment Results:**")
                st.json(alignment)
                logger.info("Performed ontology classification.")
            except Exception as e:
                st.error(f"Failed to classify across ontologies: {e}")
                logger.error(f"Error classifying across ontologies: {e}")

    # Person Search Section
    st.header("Search for a Person")
    person_query = st.text_input("Search for a person by name or affiliation")
    if person_query:
        try:
            matched_persons = st.session_state.chat_manager.topic_agent.retrieve_persons(person_query)
            if matched_persons:
                st.markdown("**Found the following persons:**")
                for p in matched_persons:
                    st.markdown(f"- {p['name']} ({p['affiliation']})")

                selected_name = st.selectbox("Select a person to see JSON-LD", [p["name"] for p in matched_persons])
                if selected_name:
                    person_data = next(x for x in matched_persons if x["name"] == selected_name)
                    jsonld = st.session_state.chat_manager.topic_agent.generate_schema_org_person(person_data)
                    st.markdown("### **Schema.org JSON-LD:**")
                    st.json(json.loads(jsonld))
                    logger.info(f"Displayed JSON-LD for person: {selected_name}")
            else:
                st.markdown("No persons found for that query.")
        except Exception as e:
            st.error(f"Person search failed: {e}")
            logger.error(f"Error searching for person: {e}")

    # Visualizations with Tabs
    st.header("Visualizations")
    tabs = st.tabs(["Knowledge Graph", "D3.js Bar Chart"])

    # Knowledge Graph Tab
    with tabs[0]:
        st.subheader("Knowledge Graph")

        graph_data = st.session_state.graph_data

        with st.expander("Graph Filters"):
            node_types = ['Topic', 'Grant', 'Patent', 'Conference', 'Person']
            selected_types = st.multiselect("Select node types to display:", node_types, default=node_types)
            layout_options = st.selectbox("Select Layout:", ["cose", "breadthfirst", "grid", "circle"])

        # Filter nodes and edges based on selected types
        try:
            filtered_nodes = [node for node in graph_data['nodes'] if node['data']['type'] in selected_types]
            filtered_edges = [
                edge for edge in graph_data['edges']
                if any(node['data']['id'] == edge['data']['source'] for node in filtered_nodes) and
                   any(node['data']['id'] == edge['data']['target'] for node in filtered_nodes)
            ]
            filtered_graph = filtered_nodes + filtered_edges
            logger.info("Filtered graph data based on selected node types.")

            # Define stylesheet for Cytoscape
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
                        'transition-duration': '0.5s',
                        'shadow-blur': '10px',
                        'shadow-color': '#333',
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

            st_cytoscapejs(
                elements=filtered_graph,
                stylesheet=stylesheet
            )
            logger.info("Rendered Knowledge Graph visualization.")
        except Exception as e:
            st.error(f"Failed to render Knowledge Graph: {e}")
            logger.error(f"Error rendering Knowledge Graph: {e}")

    # D3.js Bar Chart Visualization Tab
    with tabs[1]:
        st.subheader("D3.js Bar Chart")

        # Dynamic data for the bar chart
        dynamic_data = [
            {"name": "Topic A", "value": 30},
            {"name": "Topic B", "value": 80},
            {"name": "Topic C", "value": 45},
            {"name": "Topic D", "value": 60},
            {"name": "Topic E", "value": 20},
            {"name": "Topic F", "value": 90},
            {"name": "Topic G", "value": 55},
        ]

        # Path to the dynamic D3.js bar chart HTML file
        d3_html_path = get_asset_path("d3/bar_chart_dynamic.html")

        if d3_html_path.exists():
            try:
                d3_html = load_d3_html(d3_html_path)
                # Replace the placeholder with actual data
                d3_html = d3_html.replace("{{ data }}", json.dumps(dynamic_data))
                components.html(
                    d3_html,
                    height=550,
                    scrolling=True
                )
                logger.info("Rendered D3.js Bar Chart visualization.")
            except Exception as e:
                st.error(f"Failed to render D3.js Bar Chart: {e}")
                logger.error(f"Error rendering D3.js Bar Chart: {e}")
        else:
            st.warning("D3.js HTML file not found. Please ensure 'bar_chart_dynamic.html' exists in the 'assets/d3/' directory.")
            logger.warning(f"D3.js HTML file not found at {d3_html_path}")

if __name__ == "__main__":
    main()
