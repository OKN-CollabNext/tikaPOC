import requests
from typing import Dict, List, Any, Set
import json
from pathlib import Path
import time
from collections import defaultdict
from tqdm import tqdm

def fetch_openalex_topics() -> List[Dict[str, Any]]:
    """Fetches all topics from the OpenAlex API using cursor-based pagination."""
    base_url = "https://api.openalex.org/topics"
    topics: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = 200

    with tqdm(desc="Fetching topics") as pbar:
        while cursor:
            params = {
                "per-page": per_page,
                "cursor": cursor,
                "mailto": "chinardankhara@gmail.com"
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            topics.extend(results)
            pbar.update(len(results))
            cursor = data.get("meta", {}).get("next_cursor")
            time.sleep(0.1)  # Be kind to the API
    return topics

def validate_and_clean_topic(topic: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and cleans a single topic entry."""
    required_fields = {"id", "display_name"}
    if not all(field in topic for field in required_fields):
        missing = required_fields - set(topic.keys())
        raise ValueError(f"Topic missing required fields: {missing}")
    keywords = topic.get("keywords", [])
    if not isinstance(keywords, list):
        print(f"Warning: Invalid keywords format for topic {topic['id']}")
        keywords = []
    cleaned_keywords = {keyword.lower().strip() for keyword in keywords if isinstance(keyword, str) and keyword.strip()}
    return {
        "id": topic["id"],
        "display_name": topic["display_name"],
        "description": topic.get("description", ""),
        "keywords": list(cleaned_keywords)
    }

def process_topics(topics: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Set[str]]:
    """Process and validate all topics, returning cleaned topics and unique keywords."""
    cleaned_topics = []
    all_keywords = set()
    print("Validating and cleaning topics...")
    for topic in tqdm(topics):
        try:
            cleaned_topic = validate_and_clean_topic(topic)
            cleaned_topics.append(cleaned_topic)
            all_keywords.update(cleaned_topic["keywords"])
        except ValueError as e:
            print(f"Skipping invalid topic: {e}")
    return cleaned_topics, all_keywords

def create_keyword_mapping(topics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Creates a reverse mapping from keywords to topics."""
    from collections import defaultdict
    keyword_map = defaultdict(list)
    for topic in topics:
        topic_info = {
            "topic_name": topic["display_name"],
            "description": topic.get("description", ""),
            "topic_id": topic["id"]
        }
        for keyword in topic["keywords"]:
            keyword_map[keyword].append(topic_info)
    return dict(keyword_map)

def save_data(topics: List[Dict[str, Any]], keyword_map: Dict[str, List[Dict[str, str]]], all_keywords: Set[str], data_dir: Path) -> None:
    """Save all processed data to files."""
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "openalex_topics_raw.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    with open(data_dir / "keyword_to_topics_map.json", "w", encoding="utf-8") as f:
        json.dump(keyword_map, f, indent=2, ensure_ascii=False)
    with open(data_dir / "unique_keywords.json", "w", encoding="utf-8") as f:
        json.dump(list(all_keywords), f, indent=2, ensure_ascii=False)

def main() -> None:
    from pathlib import Path
    data_dir = Path("../../data")
    print("Fetching OpenAlex topics...")
    topics = fetch_openalex_topics()
    cleaned_topics, all_keywords = process_topics(topics)
    print("Creating keyword mapping...")
    keyword_map = create_keyword_mapping(cleaned_topics)
    save_data(cleaned_topics, keyword_map, all_keywords, data_dir)
    print("\nProcessing Summary:")
    print(f"Total topics processed: {len(cleaned_topics)}")
    print(f"Total unique keywords: {len(all_keywords)}")
    print(f"Average keywords per topic: {sum(len(t['keywords']) for t in cleaned_topics) / len(cleaned_topics):.2f}")
    print("\nSample Keywords (first 5):")
    for keyword in list(all_keywords)[:5]:
        topic_count = len(keyword_map[keyword])
        print(f"'{keyword}' appears in {topic_count} topics")

if __name__ == "__main__":
    main()
