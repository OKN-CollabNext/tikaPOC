import json
import os
import time
from typing import Any, Dict, List, Optional

import openai
from openai import AzureOpenAI
from dotenv import load_dotenv  # Import load_dotenv
from pydantic import BaseModel, Field  # Import Pydantic
import argparse
import requests
from urllib.parse import urlparse


# Load environment variables from .env file
load_dotenv()


# Define Pydantic model for the output
class OpenAlexMatch(BaseModel):
    openalex_id: str = Field(description="The OpenAlex ID of the best matching topic")


def load_topics(file_path: str) -> List[Dict[str, Any]]:
    """
    Load topics from a JSON file and flatten the hierarchical structure.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of topics with flattened hierarchy
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # If the data is from OpenAlex, it's already flat
        if file_path.endswith("openalex_topics.json"):
            return data
        
        # If it's ANZSRC data, flatten the hierarchical structure
        flattened_topics = []
        
        # Recursive function to extract all topics from the hierarchy
        def extract_topics(topics_list, flattened):
            for topic in topics_list:
                # Add this topic to the flattened list
                flattened.append(topic.copy())
                
                # If this topic has children, extract them too
                if "children" in topic and topic["children"]:
                    extract_topics(topic["children"], flattened)
        
        # Start the extraction from the top level
        extract_topics(data, flattened_topics)
        
        print(f"Loaded {len(flattened_topics)} topics from {file_path} (flattened from hierarchy)")
        
        # Print a sample of topics at each level
        level1 = [t for t in flattened_topics if len(t["code"]) == 2]
        level2 = [t for t in flattened_topics if len(t["code"]) == 4]
        level3 = [t for t in flattened_topics if len(t["code"]) == 6]
        
        print(f"Found {len(level1)} level 1 topics, {len(level2)} level 2 topics, and {len(level3)} level 3 topics")
        
        if level1:
            print(f"Sample level 1: {level1[0]['code']} - {level1[0]['label']}")
        if level2:
            print(f"Sample level 2: {level2[0]['code']} - {level2[0]['label']}")
        if level3:
            print(f"Sample level 3: {level3[0]['code']} - {level3[0]['label']}")
        
        return flattened_topics
    except Exception as e:
        print(f"Error loading topics from {file_path}: {e}")
        return []


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
    matches: Dict[str, Dict[str, Any]],
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
            print(f"Already matched ANZSRC {anzsrc_topic['code']} ({anzsrc_topic['label']}). Skipping.")
            continue

        prompt = create_match_prompt(anzsrc_topic, openalex_candidates)
        match_id = azure_openai_chat_completion(client, prompt, model)

        if match_id:
            # Find the matching OpenAlex topic to get its label
            openalex_label = ""
            for candidate in openalex_candidates:
                if candidate["id"] == match_id:
                    openalex_label = candidate["label"]
                    break
            
            # Store both ID and label information
            matches[anzsrc_topic["code"]] = {
                "anzsrc_code": anzsrc_topic["code"],
                "anzsrc_label": anzsrc_topic["label"],
                "openalex_id": match_id,
                "openalex_label": openalex_label
            }
            
            print(f"Matched ANZSRC {anzsrc_topic['code']} ({anzsrc_topic['label']}) to OpenAlex {match_id} ({openalex_label})")
        else:
            print(f"No match found for ANZSRC {anzsrc_topic['code']} ({anzsrc_topic['label']})")

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)


def fill_openalex_labels(
    matches: Dict[str, Dict[str, Any]],
    openalex_hierarchy: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Fill in the OpenAlex labels for all matches using our local OpenAlex hierarchy.
    
    Args:
        matches: Dictionary of matches with ANZSRC codes as keys
        openalex_hierarchy: The local OpenAlex hierarchy data
        
    Returns:
        Updated matches dictionary with OpenAlex labels filled in
    """
    print("\n=== Filling in OpenAlex labels ===")
    
    # Count how many labels were filled
    filled_count = 0
    
    # Iterate through all matches
    for anzsrc_code, match_data in matches.items():
        # Get the OpenAlex ID
        openalex_id = match_data.get("openalex_id")
        if not openalex_id:
            continue
        
        # Skip if the label is already filled in and not empty
        if match_data.get("openalex_label") and match_data["openalex_label"] != "":
            print(f"Label already exists for {anzsrc_code}: {match_data['openalex_label']}")
            continue
        
        # If it's a URL, extract the entity type and ID
        if openalex_id.startswith("http"):
            # Parse the URL to extract parts
            parts = openalex_id.split("/")
            if len(parts) >= 5:  # https://openalex.org/fields/11
                entity_type = parts[-2]  # e.g., 'fields', 'domains'
                entity_id = parts[-1]    # e.g., '11', '4'
                
                # Convert to our local format
                if entity_type == "domains":
                    local_id = f"domain:{entity_id}"
                elif entity_type == "fields":
                    local_id = f"field:{entity_id}"
                elif entity_type == "subfields":
                    local_id = f"subfield:{entity_id}"
                else:
                    local_id = entity_id  # For topics or other entities
                
                # Look up the label in our local hierarchy
                if local_id in openalex_hierarchy:
                    match_data["openalex_label"] = openalex_hierarchy[local_id]["label"]
                    filled_count += 1
                    print(f"Filled label for {anzsrc_code}: {match_data['openalex_label']} (from {local_id})")
                else:
                    # Try alternative formats
                    alternatives = [
                        entity_id,
                        f"{entity_type}:{entity_id}",
                        f"{entity_type[:-1]}:{entity_id}"  # Remove trailing 's'
                    ]
                    
                    found = False
                    for alt_id in alternatives:
                        if alt_id in openalex_hierarchy:
                            match_data["openalex_label"] = openalex_hierarchy[alt_id]["label"]
                            filled_count += 1
                            print(f"Filled label for {anzsrc_code}: {match_data['openalex_label']} (from alt {alt_id})")
                            found = True
                            break
                    
                    if not found:
                        # If still not found, search through all hierarchy entries
                        for hier_id, hier_data in openalex_hierarchy.items():
                            if str(hier_id).endswith(f":{entity_id}") or str(hier_id).endswith(f"/{entity_id}"):
                                match_data["openalex_label"] = hier_data["label"]
                                filled_count += 1
                                print(f"Filled label for {anzsrc_code}: {match_data['openalex_label']} (from search {hier_id})")
                                found = True
                                break
                        
                        if not found:
                            print(f"Could not find label for {anzsrc_code}: {openalex_id} (tried {local_id})")
            else:
                print(f"Invalid OpenAlex URL format for {anzsrc_code}: {openalex_id}")
        else:
            # It's not a URL, try to find it directly
            if openalex_id in openalex_hierarchy:
                match_data["openalex_label"] = openalex_hierarchy[openalex_id]["label"]
                filled_count += 1
                print(f"Filled label for {anzsrc_code}: {match_data['openalex_label']} (direct)")
            else:
                print(f"Could not find direct match for {anzsrc_code}: {openalex_id}")
    
    print(f"Filled in {filled_count} OpenAlex labels")
    return matches


def main() -> None:
    """Main function to perform the level-by-level matching."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Match ANZSRC topics to OpenAlex topics.")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], help="ANZSRC level to match (1, 2, or 3)")
    parser.add_argument("--force", action="store_true", help="Force re-matching of already matched topics")
    parser.add_argument("--fill-labels", action="store_true", help="Only fill in missing OpenAlex labels")
    args = parser.parse_args()
    
    # --- Configuration (Replace with your actual values) ---
    # Set up the Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        api_version="2023-05-15",  # Updated to a stable API version
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    )

    anzsrc_file = "data/anzsrc_for_processed.json"  # Output from previous ANZSRC script
    openalex_file = "data/openalex_topics.json"  # Output from previous OpenAlex script
    output_file = "data/matches.json"
    model_to_use = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # Use the deployment name from .env

    # --- Load Data ---
    anzsrc_topics = load_topics(anzsrc_file)
    openalex_topics = load_topics(openalex_file)
    openalex_hierarchy = build_openalex_hierarchy(openalex_topics)

    # --- Load existing matches if file exists ---
    matches: Dict[str, Dict[str, Any]] = {}  # Changed from Dict[str, str] to Dict[str, Dict[str, Any]]
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_matches = json.load(f)
                
                # Convert old format (if needed)
                for key, value in existing_matches.items():
                    if isinstance(value, str):
                        # Find ANZSRC label
                        anzsrc_label = ""
                        for topic in anzsrc_topics:
                            if topic["code"] == key:
                                anzsrc_label = topic["label"]
                                break
                        
                        # Store in new format
                        matches[key] = {
                            "anzsrc_code": key,
                            "anzsrc_label": anzsrc_label,
                            "openalex_id": value,
                            "openalex_label": ""  # Empty label to be filled later
                        }
                    else:
                        # Already in new format
                        matches[key] = value
                
            print(f"Loaded {len(matches)} existing matches from {output_file}")
        except json.JSONDecodeError:
            print(f"Error loading {output_file}. Starting with empty matches.")
    
    # If only filling labels, do that and exit
    if args.fill_labels:
        matches = fill_openalex_labels(matches, openalex_hierarchy)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=4)
        print(f"Updated matches with OpenAlex labels saved to {output_file}")
        return

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
        
        # Fill in any missing OpenAlex labels
        matches = fill_openalex_labels(matches, openalex_hierarchy)
        
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
        
        # Fill in any missing OpenAlex labels
        matches = fill_openalex_labels(matches, openalex_hierarchy)
        
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
        
        # Fill in any missing OpenAlex labels
        matches = fill_openalex_labels(matches, openalex_hierarchy)
        
        # Save after level 3
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=4)
        print(f"Level 3 matching results saved to {output_file}")


if __name__ == "__main__":
    main()