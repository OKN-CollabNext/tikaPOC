import networkx as nx
from pyvis.network import Network

def create_topic_graph(topics: list) -> str:
    """
    Create an interactive HTML network graph of topics.
    Each node represents a topic and edges connect topics sharing keywords.
    """
    G = nx.Graph()

    # Add nodes
    for topic in topics:
        G.add_node(topic['id'], label=topic['display_name'], title=topic.get('description', ''))

    # Connect nodes if they share a common keyword (naïve pairwise check)
    for i, topic1 in enumerate(topics):
        for topic2 in topics[i+1:]:
            common = set(topic1.get('keywords', [])) & set(topic2.get('keywords', []))
            if common:
                G.add_edge(topic1['id'], topic2['id'], weight=len(common))

    # Use PyVis for interactive visualization
    net = Network(height='600px', width='100%', notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])

    # Save and return the HTML file path
    html_path = "topic_graph.html"
    net.save_graph(html_path)
    return html_path
