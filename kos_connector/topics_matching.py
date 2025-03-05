import json
import os
import time
from typing import Any, Dict, List, Optional

import openai
from openai import AzureOpenAI


def load_topics(filename: str) -> List[Dict[str, Any]]:
    """Loads topics from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def build_openalex_hierarchy(
    openalex_topics: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Builds a hierarchical dictionary from the OpenAlex topics."""
    hierarchy: Dict[str, Dict[str, Any]] = {}
    # Create a lookup by ID
    id_to_topic: Dict[str, Dict[str, Any]] = {
        topic["id"]: topic for topic in openalex_topics
    }

    for topic in openalex_topics:
        topic_id: str = topic["id"]
        hierarchy[topic_id] = {
            "label": topic["display_name"],
            "level": topic["level"],
            "children": [],  # Initialize children list
        }
        if topic["ancestors"]:
            parent_id = topic["ancestors"][-1][
                "id"
            ]  # immediate parent is the last ancestor
            if parent_id in id_to_topic:
                if "children" not in id_to_topic[parent_id]:
                    id_to_topic[parent_id]["children"] = []
                id_to_topic[parent_id]["children"].append(
                    topic_id
                )  # add to parent's children

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
) -> str:
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
        )
        return response.choices[0].message.content.strip()
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e}.  Waiting 60 seconds...")
        time.sleep(60)
        return azure_openai_chat_completion(
            client, prompt, model, max_tokens, temperature, stop
        )  # Retry
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def create_match_prompt(
    anzsrc_topic: Dict[str, Any], openalex_topics: List[Dict[str, str]]
) -> str:
    """Creates the prompt for the LLM matching."""
    prompt = (
        "Match the following ANZSRC topic to the MOST relevant OpenAlex topic ID.\n\n"
        "ANZSRC Topic:\n"
        f"Code: {anzsrc_topic['code']}\n"
        f"Label: {anzsrc_topic['label']}\n\n"
        "OpenAlex Topics (ID: Label):\n"
    )
    for topic in openalex_topics:
        prompt += f"{topic['id']}: {topic['label']}\n"
    prompt += "\nReturn ONLY the OpenAlex ID of the best match, nothing else. For example: 'X123456789'"
    return prompt


def match_topics(
    client: AzureOpenAI,
    anzsrc_topics: List[Dict[str, Any]],
    openalex_hierarchy: Dict[str, Dict[str, Any]],
    anzsrc_level: int,
    openalex_levels: List[int],
    model: str,
    matches: Dict[str, str],
    parent_match: str = "",
) -> None:
    """Recursively matches ANZSRC topics to OpenAlex topics."""

    # Filter ANZSRC topics by level.  Handle level 1 differently
    if anzsrc_level == 1:
        current_anzsrc_topics = anzsrc_topics
    else:
        current_anzsrc_topics = [
            t for t in anzsrc_topics if t["code"].startswith(parent_match)
        ]

    if not current_anzsrc_topics:
        return  # No more topics at this level

    openalex_candidates = get_openalex_topics_by_level(
        openalex_hierarchy, openalex_levels
    )
    if not openalex_candidates:
        print(
            f"No OpenAlex candidates found for levels {openalex_levels}. Skipping."
        )
        return

    for anzsrc_topic in current_anzsrc_topics:
        prompt = create_match_prompt(anzsrc_topic, openalex_candidates)
        match_id = azure_openai_chat_completion(client, prompt, model)

        if match_id:
            matches[anzsrc_topic["code"]] = match_id
            print(f"Matched ANZSRC {anzsrc_topic['code']} to OpenAlex {match_id}")

            # Recursive call for next level
            if anzsrc_level < 3:  # We have more levels to process
                next_anzsrc_level = anzsrc_level + 1
                next_openalex_levels = [
                    openalex_levels[-1] + 1
                ]  # Next OpenAlex level

                match_topics(
                    client,
                    anzsrc_topics,
                    openalex_hierarchy,
                    next_anzsrc_level,
                    next_openalex_levels,
                    model,
                    matches,
                    anzsrc_topic["code"],
                )
        else:
            print(f"No match found for ANZSRC {anzsrc_topic['code']}")


def main() -> None:
    """Main function to perform the multi-level matching."""
    # --- Configuration (Replace with your actual values) ---
    # Set up the Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    )

    anzsrc_file = "anzsrc_for_processed.json"  # Output from previous ANZSRC script
    openalex_file = "openalex_topics.json"  # Output from previous OpenAlex script
    output_file = "matches.json"
    model_to_use = "gpt-4o"  # Or your specific deployment name

    # --- Load Data ---
    anzsrc_topics = load_topics(anzsrc_file)
    openalex_topics = load_topics(openalex_file)
    openalex_hierarchy = build_openalex_hierarchy(openalex_topics)

    # --- Perform Matching ---
    matches: Dict[str, str] = {}
    match_topics(
        client,
        anzsrc_topics,
        openalex_hierarchy,
        anzsrc_level=1,
        openalex_levels=[0, 1],  # Start with OpenAlex levels 0 and 1
        model=model_to_use,
        matches=matches,
    )

    # --- Save Results ---
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=4)
    print(f"Matching results saved to {output_file}")


if __name__ == "__main__":
    main()