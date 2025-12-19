"""Module for processing scene graph data files."""

import json
from pathlib import Path
from typing import List, Dict, Any
import random
from tqdm import tqdm
from .config import MAX_ITEMS_PER_SCENE, MAX_ATTRIBUTES_PER_ITEM


def process_json_file(data: dict) -> str:
    """Process a single JSON file and extract scene graph data for LLM prompting.

    Args:
        file_path: Path to the JSON file containing scene graph data

    Returns:
        Formatted prompt string for LLM processing
    """

    desc_prompt = """For the given scene graph, come up with a positive and a negative general context in the following format:

{{
  "positive": {{
    "description": "[A description of the scene graph using the relationships and the original attributes fields]",
    "context": "[A brief statement of the general context of the description (6-12 words)]"
  }},
  "negative": {{
    "description": "[A description of the scene graph using the relationships and the negative attributes field.]",
    "context": "[A brief statement of the general context of the description (6-12 words)]"
  }}
}}
---
**Example 1**
Input:
{{
"items": [
{{
"object_id": 0,
"category": "conductor",
"original attributes": [
"concentrated",
"actively",
"waving",
"black"
]
}},
{{
"object_id": 1,
"category": "baton",
"original attributes": [
"wooden",
"long",
"thin"
]
}}
],
"relationships": [
{{
"subject": 0,
"relation": "holding",
"object": 1
}},
{{
"subject": 0,
"relation": "waving",
"object": 1
}}
]
}}
Output:
{{
  "positive": {{
    "description": "The conductor is concentrated and actively waving the baton.",
    "context": "A conductor in the middle of the concert."
  }},
  "negative": {{
    "description": "The conductor is idle and not waving the baton.",
    "context": "A conductor after the concert."
  }}
}}
---
**Example 2**
Input:
{{
  "items": [
    {{
      "object_id": 0,
      "category": "horse",
      "original attributes": [
        "single/one/individual/sole",
        "vertical/upright/standing"
      ],
      "negative attributes": [
        "leather"
      ]
    }},
    {{
      "object_id": 1,
      "category": "person",
      "original attributes": [
        "male/man/guy/boy",
        "adult/old/aged"
      ],
      "negative attributes": [
        "female/woman/girl"
      ]
    }},
    {{
      "object_id": 2,
      "category": "person",
      "original attributes": [
        "young/baby"
      ],
      "negative attributes": [
        "female/woman/girl",
        "male/man/guy/boy"
      ]
    }}
  ],
  "relationships": [
    {{
      "subject": 0,
      "relation": "near",
      "object": 1
    }},
    {{
      "subject": 1,
      "relation": "holding",
      "object": 2
    }}
  ]
}}
Output:
{{
  "positive": {{
    "description": "A male adult holding a young baby near a still horse.",
    "context": "A father letting his baby pet a horse's nose."
  }},
  "negative": {{
    "description": "A girl holding an old man far from some running horses.",
    "context": "A girl watching a horse race with her grandfather."
        }}
}}
---
**Example 3**
Input:
{{
  "items": [
    {{
      "object_id": 0,
      "category": "bowl",
      "original attributes": [
        "ceramic/brick/porcelain"
      ],
      "negative attributes": [
        "short"
      ]
    }},
    {{
      "object_id": 1,
      "category": "apple",
      "original attributes": [
        "full/whole"
      ],
      "negative attributes": [
        "ordered/arranged/organized/tidy",
        "big/large/giant/huge",
        "small/little/tiny"
      ]
    }}
  ],
  "relationships": [
    {{
      "subject": 0,
      "relation": "contains",
      "object": 1
    }}
  ]
}}

Output:
{{
  "positive": {{
    "description": "A ceramic, porcelain bowl filled with fresh, whole apples.",
    "context": "A bowl of apples is being served.",
  }},
  "negative": {{
    "description": "A ceramic, porcelain bowl filled with cookie crumbs.",
    "context": "A bowl of cookies has been finished.",
  }}
}}
---

Input:
{sg}
Output:
"""

    # Limit to MAX_ITEMS_PER_SCENE items and filter attributes
    curr = {"items": [], "relationships": []}
    items = data.get('annotations', [])
    items = random.sample(items, min(len(items), MAX_ITEMS_PER_SCENE))
    print("Selected objects:", [item.get('category', '') for item in items])
    for item in items:
        pos_attributes = item.get('original attributes', [])
        neg_attributes = item.get('five negative attributes', [])
        bad_attributes = ["clean/neat", "messy/disordered/unordered/disorganized/unorganized/cluttered/untidy"]
        for attr in bad_attributes:
            if attr in neg_attributes:
                neg_attributes.remove(attr)
        curr['items'].append({
            "object_id": item.get('object_id'),
            "category": item.get('category', ''),
            "original attributes": pos_attributes[:min(len(pos_attributes), MAX_ATTRIBUTES_PER_ITEM)],
            "negative attributes": random.sample(neg_attributes, min(len(neg_attributes), MAX_ATTRIBUTES_PER_ITEM))
        })

        if "relationship_final_id" in item:
            curr['relationships'].append(item['relationship_final_id'])

    return desc_prompt.format(sg=json.dumps(curr, indent=2))


def process_data(file_name: str, sample=None) -> List[Dict[str, str]]:
    """Process all or a subset of items from the given file and return a list of prompts."""
    with open(file_name, 'r') as f:
        data = json.load(f)['data']
    if sample is not None:
        data = random.sample(data, sample)
    
    prompts = []
    img_filenames = []
    for img_data in tqdm(data, desc="Gathering description prompts"):
        try:
            question = process_json_file(img_data)
            prompt = (
                "SYSTEM\nYou are a helpful assistant.\n"
                f"USER\n/{question}\n"
                f"ASSISTANT\n"
            )
            inputs = {
                "prompt": prompt
            }
            prompts.append(inputs)
            img_filenames.append({"filename": img_data['file_name']})
        except Exception as e:
            print(f"Error processing {img_data['file_name']}: {str(e)}")

    return prompts, img_filenames