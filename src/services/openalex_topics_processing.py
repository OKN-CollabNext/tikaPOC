import requests
from typing import Dict, List, Any, Set, Tuple
import json
from pathlib import Path
import time
from collections import defaultdict
from tqdm import tqdm
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def fetch_openalex_topics() -> List[Dict[str, Any]]:
    """Fetches all topics from OpenAlex API using cursor-based pagination."""
    base_url = "https://api.openalex.org/topics"
    topics: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = 200

    with tqdm(desc="Fetching OpenAlex topics") as pbar:
        while cursor:
            params = {
                "per-page": per_page,
                "cursor": cursor,
                "mailto": "chinardankhara@gmail.com",
            }
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    logger.info("No more topics fetched from OpenAlex.")
                    break

                topics.extend(results)
                pbar.update(len(results))

                cursor = data.get("meta", {}).get("next_cursor")
                time.sleep(0.1)  # Be polite to the API
            except requests.RequestException as e:
                logger.error(f"Error fetching topics from OpenAlex: {e}")
                break

    logger.info(f"Fetched {len(topics)} topics from OpenAlex.")
    return topics


def validate_and_clean_topic(topic: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and cleans a single topic entry."""
    required_fields = {"id", "display_name"}

    # Check required fields
    if not required_fields.issubset(topic.keys()):
        missing = required_fields - set(topic.keys())
        logger.warning(f"Topic missing required fields: {missing}. Skipping.")
        raise ValueError(f"Topic missing required fields: {missing}")

    # Clean and validate keywords
    keywords = topic.get("keywords", [])
    if not isinstance(keywords, list):
        logger.warning(f"Invalid keywords format for topic {topic['id']}. Setting empty list.")
        keywords = []

    # Clean keywords: lowercase, remove empty strings, strip whitespace
    cleaned_keywords = {
        keyword.lower().strip()
        for keyword in keywords
        if isinstance(keyword, str) and keyword.strip()
    }

    cleaned_topic = {
        "id": topic["id"],
        "display_name": topic["display_name"],
        "description": topic.get("description", ""),
        "keywords": list(cleaned_keywords),
    }
    logger.debug(f"Cleaned topic: {cleaned_topic['id']}")
    return cleaned_topic


def process_topics(
    topics: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Process and validate all topics, returning cleaned topics and unique keywords."""
    cleaned_topics = []
    all_keywords = set()

    logger.info("Validating and cleaning fetched topics.")
    for topic in tqdm(topics, desc="Processing topics"):
        try:
            cleaned_topic = validate_and_clean_topic(topic)
            cleaned_topics.append(cleaned_topic)
            all_keywords.update(cleaned_topic["keywords"])
        except ValueError:
            continue

    logger.info(f"Processed {len(cleaned_topics)} valid topics with {len(all_keywords)} unique keywords.")
    return cleaned_topics, all_keywords


def create_keyword_mapping(
    topics: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, str]]]:
    """Creates a reverse mapping from keywords to topics."""
    keyword_map = defaultdict(list)

    logger.info("Creating keyword to topics mapping.")
    for topic in topics:
        topic_info = {
            "topic_name": topic["display_name"],
            "description": topic.get("description", ""),
            "topic_id": topic["id"],
        }

        for keyword in topic["keywords"]:
            keyword_map[keyword].append(topic_info)

    logger.info("Keyword to topics mapping created.")
    return dict(keyword_map)


def save_data(
    topics: List[Dict[str, Any]],
    keyword_map: Dict[str, List[Dict[str, str]]],
    all_keywords: Set[str],
    data_dir: Path,
) -> None:
    """Save all processed data to files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving data to directory: {data_dir}")

    # Save raw topics
    with open(data_dir / "openalex_topics_raw.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    logger.info("Saved raw topics to openalex_topics_raw.json.")

    # Save keyword mapping
    with open(data_dir / "keyword_to_topics_map.json", "w", encoding="utf-8") as f:
        json.dump(keyword_map, f, indent=2, ensure_ascii=False)
    logger.info("Saved keyword mapping to keyword_to_topics_map.json.")

    # Save unique keywords list
    with open(data_dir / "unique_keywords.json", "w", encoding="utf-8") as f:
        json.dump(list(all_keywords), f, indent=2, ensure_ascii=False)
    logger.info("Saved unique keywords to unique_keywords.json.")


def main() -> None:
    data_dir = Path("../../data")

    # Fetch topics
    logger.info("Starting to fetch topics from OpenAlex.")
    topics = fetch_openalex_topics()

    # Process and validate topics
    cleaned_topics, all_keywords = process_topics(topics)

    # Create keyword mapping
    keyword_map = create_keyword_mapping(cleaned_topics)

    # Save all data
    save_data(cleaned_topics, keyword_map, all_keywords, data_dir)

    # Print summary
    logger.info("\nProcessing Summary:")
    logger.info(f"Total topics processed: {len(cleaned_topics)}")
    logger.info(f"Total unique keywords: {len(all_keywords)}")
    avg_keywords = sum(len(t['keywords']) for t in cleaned_topics) / len(cleaned_topics) if cleaned_topics else 0
    logger.info(f"Average keywords per topic: {avg_keywords:.2f}")

    # Sample validation
    logger.info("\nSample Keywords (first 5):")
    for keyword in list(all_keywords)[:5]:
        topic_count = len(keyword_map[keyword])
        logger.info(f"'{keyword}' appears in {topic_count} topics")


if __name__ == "__main__":
    main()
