import json
import os
import time
from typing import Any, Dict, List, Optional

import openai
from openai import AzureOpenAI
from dotenv import load_dotenv  # Import load_dotenv
from pydantic import BaseModel, Field  # Import Pydantic
import argparse


# Load environment variables from .env file
load_dotenv()


# Define Pydantic model for the output
class OpenAlexMatch(BaseModel):
    openalex_id: str = Field(description="The OpenAlex ID of the best matching topic")


def load_topics(filename: str) -> List[Dict[str, Any]]:
    """Loads topics from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def build_openalex_hierarchy(
    openalex_topics: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Builds a hierarchical dictionary from the OpenAlex topics.
    Each topic in OpenAlex belongs to a hierarchy: domain > field > subfield > topic
    """
    hierarchy: Dict[str, Dict[str, Any]] = {}
    
    # First pass: add all topics to the hierarchy
    for topic in openalex_topics:
        topic_id: str = topic["id"]
        hierarchy[topic_id] = {
            "id": topic_id,
            "label": topic["display_name"],
            "works_count": topic["works_count"],
            "children": [],
            "level": 3  # Topics are at level 3 (0-based: domain=0, field=1, subfield=2, topic=3)
        }
        
        # Add domain if not exists
        domain_id = f"domain:{topic['domain']['id']}"
        if domain_id not in hierarchy:
            hierarchy[domain_id] = {
                "id": domain_id,
                "label": topic['domain']['display_name'],
                "works_count": 10000000,  # Arbitrary high number to ensure domains are considered top-level
                "children": [],
                "level": 0  # Domain is level 0 (highest level)
            }
            
        # Add field if not exists
        field_id = f"field:{topic['field']['id']}"
        if field_id not in hierarchy:
            hierarchy[field_id] = {
                "id": field_id,
                "label": topic['field']['display_name'],
                "works_count": 5000000,  # High number but less than domain
                "children": [],
                "level": 1  # Field is level 1
            }
            
        # Add subfield if not exists
        subfield_id = f"subfield:{topic['subfield']['id']}"
        if subfield_id not in hierarchy:
            hierarchy[subfield_id] = {
                "id": subfield_id,
                "label": topic['subfield']['display_name'],
                "works_count": 1000000,  # High number but less than field
                "children": [],
                "level": 2  # Subfield is level 2
            }
            
        # Build parent-child relationships
        # Add topic to subfield's children
        if topic_id not in hierarchy[subfield_id]["children"]:
            hierarchy[subfield_id]["children"].append(topic_id)
            
        # Add subfield to field's children
        if subfield_id not in hierarchy[field_id]["children"]:
            hierarchy[field_id]["children"].append(subfield_id)
            
        # Add field to domain's children
        if field_id not in hierarchy[domain_id]["children"]:
            hierarchy[domain_id]["children"].append(field_id)
    
    return hierarchy


def get_openalex_topics_by_level(
    hierarchy: Dict[str, Dict[str, Any]], levels: List[int]
) -> List[Dict[str, str]]:
    """Gets OpenAlex topics at specified levels."""
    return [
        {"id": topic_id, "label": topic_data["label"]}
        for topic_id, topic_data in hierarchy.items()
        if topic_data["level"] in levels
    ]


def get_children_labels(
    hierarchy: Dict[str, Dict[str, Any]], parent_id: str
) -> List[Dict[str, str]]:
    """Gets the labels of child topics in the hierarchy."""
    if parent_id not in hierarchy or "children" not in hierarchy[parent_id]:
        return []
    return [
        {"id": child_id, "label": hierarchy[child_id]["label"]}
        for child_id in hierarchy[parent_id]["children"]
    ]


def azure_openai_chat_completion(
    client: AzureOpenAI,
    prompt: str,
    model: str,
    max_tokens: int = 50,
    temperature: float = 0.2,
    stop: Optional[List[str]] = None,
) -> Optional[str]:
    """Gets a chat completion from Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            response_format={"type": "json_object"},  # Enforce structured output without schema
        )
        
        # Parse the JSON response
        response_content = response.choices[0].message.content
        if response_content:
            try:
                json_response = json.loads(response_content)
                return json_response.get("openalex_id")  # Extract openalex_id
            except json.JSONDecodeError:
                print("Invalid JSON response received.")
                return None
        return None

    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e}.  Waiting 60 seconds...")
        time.sleep(60)
        return azure_openai_chat_completion(
            client, prompt, model, max_tokens, temperature, stop
        )  # Retry
    except openai.APIConnectionError as e:
        print(f"Connection error: {e}")
        return None
    except openai.BadRequestError as e:
        print(f"Bad request error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_match_prompt(
    anzsrc_topic: Dict[str, Any], openalex_topics: List[Dict[str, str]]
) -> str:
    """Creates the prompt for the LLM matching."""
    prompt = (
        "Match the following ANZSRC topic to the MOST relevant OpenAlex topic ID.\n"
        "Return the result as a JSON object with the key 'openalex_id'.\n\n" # Modified prompt
        "ANZSRC Topic:\n"
        f"Code: {anzsrc_topic['code']}\n"
        f"Label: {anzsrc_topic['label']}\n\n"
        "OpenAlex Topics (ID: Label):\n"
    )
    for topic in openalex_topics:
        prompt += f"{topic['id']}: {topic['label']}\n"
    prompt += "\nReturn ONLY a JSON object with the OpenAlex ID of the best match. For example: {\"openalex_id\": \"X123456789\"}" # Modified prompt
    return prompt


def match_topics_by_level(
    client: AzureOpenAI,
    anzsrc_topics: List[Dict[str, Any]],
    openalex_hierarchy: Dict[str, Dict[str, Any]],
    anzsrc_level: int,
    openalex_levels: List[int],
    model: str,
    matches: Dict[str, str],
) -> None:
    """
    Matches ANZSRC topics at a specific level to OpenAlex topics.
    This is a non-recursive version that processes one level at a time.
    """
    # Filter ANZSRC topics by level
    if anzsrc_level == 1:
        # Level 1 topics have 2-digit codes
        current_anzsrc_topics = [t for t in anzsrc_topics if len(t["code"]) == 2]
    elif anzsrc_level == 2:
        # Level 2 topics have 4-digit codes
        current_anzsrc_topics = [t for t in anzsrc_topics if len(t["code"]) == 4]
    elif anzsrc_level == 3:
        # Level 3 topics have 6-digit codes
        current_anzsrc_topics = [t for t in anzsrc_topics if len(t["code"]) == 6]
    else:
        print(f"Invalid ANZSRC level: {anzsrc_level}")
        return

    if not current_anzsrc_topics:
        print(f"No ANZSRC topics found at level {anzsrc_level}. Skipping.")
        return

    openalex_candidates = get_openalex_topics_by_level(
        openalex_hierarchy, openalex_levels
    )
    if not openalex_candidates:
        print(
            f"No OpenAlex candidates found for levels {openalex_levels}. Skipping."
        )
        return

    for anzsrc_topic in current_anzsrc_topics:
        # Skip if we already have a match for this topic
        if anzsrc_topic["code"] in matches:
            print(f"Already matched ANZSRC {anzsrc_topic['code']}. Skipping.")
            continue

        prompt = create_match_prompt(anzsrc_topic, openalex_candidates)
        match_id = azure_openai_chat_completion(client, prompt, model)

        if match_id:
            matches[anzsrc_topic["code"]] = match_id
            print(f"Matched ANZSRC {anzsrc_topic['code']} to OpenAlex {match_id}")
        else:
            print(f"No match found for ANZSRC {anzsrc_topic['code']}")

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)


def main() -> None:
    """Main function to perform the level-by-level matching."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Match ANZSRC topics to OpenAlex topics.")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], help="ANZSRC level to match (1, 2, or 3)")
    parser.add_argument("--force", action="store_true", help="Force re-matching of already matched topics")
    args = parser.parse_args()
    
    # --- Configuration (Replace with your actual values) ---
    # Set up the Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    )

    anzsrc_file = "data/anzsrc_for_processed.json"  # Output from previous ANZSRC script
    openalex_file = "data/openalex_topics.json"  # Output from previous OpenAlex script
    output_file = "data/matches.json"
    model_to_use = "gpt-4o"  # Or your specific deployment name

    # --- Load Data ---
    anzsrc_topics = load_topics(anzsrc_file)
    openalex_topics = load_topics(openalex_file)
    openalex_hierarchy = build_openalex_hierarchy(openalex_topics)

    # --- Load existing matches if file exists ---
    matches: Dict[str, str] = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                matches = json.load(f)
            print(f"Loaded {len(matches)} existing matches from {output_file}")
        except json.JSONDecodeError:
            print(f"Error loading {output_file}. Starting with empty matches.")

    # If force flag is set, clear existing matches for the specified level
    if args.force and args.level:
        if args.level == 1:
            matches = {k: v for k, v in matches.items() if len(k) != 2}
            print("Cleared existing level 1 matches.")
        elif args.level == 2:
            matches = {k: v for k, v in matches.items() if len(k) != 4}
            print("Cleared existing level 2 matches.")
        elif args.level == 3:
            matches = {k: v for k, v in matches.items() if len(k) != 6}
            print("Cleared existing level 3 matches.")

    # --- Perform matching based on command line arguments ---
    if args.level == 1 or args.level is None:
        print("\n=== Matching ANZSRC Level 1 to OpenAlex Levels 0,1 ===")
        match_topics_by_level(
            client,
            anzsrc_topics,
            openalex_hierarchy,
            anzsrc_level=1,
            openalex_levels=[0, 1],  # Domain and Field levels
            model=model_to_use,
            matches=matches,
        )
        
        # Save after level 1
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=4)
        print(f"Level 1 matching results saved to {output_file}")
        
        # If specific level was requested, exit
        if args.level == 1:
            return

    if args.level == 2 or args.level is None:
        print("\n=== Matching ANZSRC Level 2 to OpenAlex Level 2 ===")
        match_topics_by_level(
            client,
            anzsrc_topics,
            openalex_hierarchy,
            anzsrc_level=2,
            openalex_levels=[2],  # Subfield level
            model=model_to_use,
            matches=matches,
        )
        
        # Save after level 2
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=4)
        print(f"Level 2 matching results saved to {output_file}")
        
        # If specific level was requested, exit
        if args.level == 2:
            return

    if args.level == 3 or args.level is None:
        print("\n=== Matching ANZSRC Level 3 to OpenAlex Level 3 ===")
        match_topics_by_level(
            client,
            anzsrc_topics,
            openalex_hierarchy,
            anzsrc_level=3,
            openalex_levels=[3],  # Topic level
            model=model_to_use,
            matches=matches,
        )
        
        # Save after level 3
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=4)
        print(f"Level 3 matching results saved to {output_file}")


if __name__ == "__main__":
    main()