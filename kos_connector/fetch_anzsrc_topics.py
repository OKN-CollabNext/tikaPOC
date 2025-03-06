import json
import os
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_anzsrc_page(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a page of ANZSRC data from the API.
    
    Args:
        url: The URL to fetch data from
        
    Returns:
        The JSON response as a dictionary, or None if the request failed
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def process_anzsrc_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process ANZSRC items to extract relevant information.
    
    Args:
        items: List of ANZSRC items from the API
        
    Returns:
        List of processed ANZSRC topics
    """
    processed_items = []
    
    for item in items:
        # Extract the code (notation)
        code = item.get("notation")
        if not code:
            continue
            
        # Extract the label
        label = None
        pref_label = item.get("prefLabel")
        if pref_label and isinstance(pref_label, dict):
            label = pref_label.get("_value")
        
        # Extract the definition
        definition = item.get("definition")
        
        # Extract broader concept (parent)
        broader = item.get("broader")
        
        # Extract narrower concepts (children)
        narrower = item.get("narrower", [])
        
        # Create processed item
        processed_item = {
            "code": code,
            "label": label,
            "definition": definition,
            "broader": broader,
            "narrower": narrower
        }
        
        processed_items.append(processed_item)
    
    return processed_items

def fetch_all_anzsrc_topics() -> List[Dict[str, Any]]:
    """
    Fetch all ANZSRC topics by traversing through all pages.
    
    Returns:
        List of all ANZSRC topics
    """
    # Start with the first page
    base_url = "http://vocabs.ardc.edu.au/repository/api/lda/anzsrc-2020-for/concept.json"
    page_size = 1500  # Maximum page size
    url = f"{base_url}?_pageSize={page_size}"
    
    all_topics = []
    page_num = 0
    
    while url:
        print(f"Fetching page {page_num}...")
        data = fetch_anzsrc_page(url)
        
        if not data or "result" not in data:
            print(f"No valid data found at {url}")
            break
            
        result = data["result"]
        
        # Process items on this page
        if "items" in result:
            items = process_anzsrc_items(result["items"])
            all_topics.extend(items)
            print(f"Processed {len(items)} items from page {page_num}")
        
        # Check if there's a next page
        next_page = None
        if "next" in result:
            next_page = result["next"]
        
        # Update URL for next iteration
        if next_page:
            url = next_page
            page_num += 1
        else:
            url = None
            print("No more pages to fetch")
    
    return all_topics

def main():
    """Main function to fetch and save ANZSRC topics."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Fetch all ANZSRC topics
    print("Fetching all ANZSRC topics...")
    all_topics = fetch_all_anzsrc_topics()
    
    # Save to file
    output_file = "data/anzsrc_for_complete.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_topics, f, indent=2)
    
    print(f"Saved {len(all_topics)} ANZSRC topics to {output_file}")
    
    # Process the topics to create a hierarchical structure
    print("Processing topics to create hierarchical structure...")
    
    # Create a dictionary to map codes to their corresponding topics
    topics_by_code = {topic["code"]: topic for topic in all_topics}
    
    # Create a hierarchical structure
    hierarchical_topics = []
    
    # Find top-level topics (those without a broader concept)
    for topic in all_topics:
        if not topic.get("broader"):
            # This is a top-level topic
            hierarchical_topic = process_topic_hierarchy(topic, topics_by_code)
            hierarchical_topics.append(hierarchical_topic)
    
    # Save hierarchical structure to file
    hierarchical_output_file = "data/anzsrc_for_hierarchical.json"
    with open(hierarchical_output_file, "w", encoding="utf-8") as f:
        json.dump(hierarchical_topics, f, indent=2)
    
    print(f"Saved hierarchical structure to {hierarchical_output_file}")
    
    # Also save the processed flat structure for compatibility with existing code
    processed_output_file = "data/anzsrc_for_processed.json"
    with open(processed_output_file, "w", encoding="utf-8") as f:
        json.dump(all_topics, f, indent=2)
    
    print(f"Saved processed topics to {processed_output_file}")

def process_topic_hierarchy(topic: Dict[str, Any], topics_by_code: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process a topic to include its full hierarchy.
    
    Args:
        topic: The topic to process
        topics_by_code: Dictionary mapping codes to topics
        
    Returns:
        The topic with its children processed recursively
    """
    # Create a copy of the topic to avoid modifying the original
    processed_topic = topic.copy()
    
    # Process children
    children = []
    for narrower_code in topic.get("narrower", []):
        # Extract the code from the URI
        if isinstance(narrower_code, str) and "/" in narrower_code:
            code = narrower_code.split("/")[-1]
            
            # Find the child topic
            child_topic = topics_by_code.get(code)
            if child_topic:
                # Process the child recursively
                processed_child = process_topic_hierarchy(child_topic, topics_by_code)
                children.append(processed_child)
    
    # Replace narrower with processed children
    if children:
        processed_topic["children"] = children
    
    return processed_topic

if __name__ == "__main__":
    main() 