"""Helpers for scene graph evaluation prompts and parsing."""

import json
import os
import re
from typing import Any, Dict, List, Optional

def extract_scene_graph(prompt_text):
    # 1. Isolate the "Current Task" section
    # We split by 'Current Task:' and take the last part
    if "Current Task:" in prompt_text:
        current_task_text = prompt_text.split("Current Task:")[-1]
    else:
        # Fallback if just the raw data is passed
        current_task_text = prompt_text

    # 2. Extract Objects and Relationships blocks
    # We look for the text between specific headers
    try:
        # Get everything after "Objects:" and before "Relationships:"
        obj_section = re.search(r'Objects:(.*?)Relationships:', current_task_text, re.DOTALL).group(1)
        
        # Get everything after "Relationships:" and before the next section 
        # (usually [Step-by-Step Reasoning] or end of string)
        rel_section_match = re.search(r'Relationships:(.*?)(?:\[Step-by-Step Reasoning\]|$)', current_task_text, re.DOTALL)
        rel_section = rel_section_match.group(1)
    except AttributeError:
        return {"error": "Could not find Objects or Relationships sections in the expected format."}

    # 3. Parse Objects
    objects = []
    current_obj = None

    # Regex to identify a new object line: "- 1 person (object id : 4)"
    # Captures: 1=count (unused), 2=name, 3=id
    obj_pattern = re.compile(r'-\s*\d+\s+(.+?)\s*\(object id\s*:\s*(\d+)\)')
    
    # Regex for attributes: "-clothes color:black" or "-cozy"
    attr_pattern = re.compile(r'-\s*(.+)')

    for line in obj_section.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this line defines a new object
        obj_match = obj_pattern.match(line)
        if obj_match:
            # Save previous object if exists
            if current_obj:
                objects.append(current_obj)
            
            # Start new object
            current_obj = {
                "id": int(obj_match.group(2)),
                "name": obj_match.group(1).strip(),
                "attributes": []
            }
        
        # If it's not a new object, it's an attribute for the current object
        elif current_obj:
            attr_match = attr_pattern.match(line)
            if attr_match:
                raw_attr = attr_match.group(1).strip()
                current_obj["attributes"].append(raw_attr)

    # Append the last object
    if current_obj:
        objects.append(current_obj)

    # 4. Parse Relationships
    relationships = []
    # Regex: "- Object 12 with Object 110"
    # Captures: 1=Subject ID, 2=Relation, 3=Object ID
    rel_pattern = re.compile(r'-\s*Object\s*(\d+)\s+(.+?)\s+Object\s*(\d+)')

    for line in rel_section.strip().split('\n'):
        line = line.strip()
        rel_match = rel_pattern.match(line)
        if rel_match:
            relationships.append({
                "subject": int(rel_match.group(1)),
                "relation": rel_match.group(2).strip(),
                "object": int(rel_match.group(3))
            })

    return {
        "objects": objects,
        "relations": relationships
    }

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def build_object_prompt(image_width: int, image_height: int, entity: Dict[str, Any]) -> str:
    attrs = entity.get("attributes") or []
    return (
        "SYSTEM\nYou are a helpful assistant.\nUSER\n"
        "You are evaluating a generated image against a scene graph.\n"
        "Return ONLY valid JSON with no extra commentary.\n\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n" 
        f"Image size: width={image_width}, height={image_height}.\n\n"
        "Task: Determine if the object is visible and return its bounding box.\n"
        f"Object: id={entity.get('id')}, name={entity.get('name')}, attributes={json.dumps(attrs)}\n\n"
        "Output JSON format:\n"
        "{\n"
        "  \"id\": 0,\n"
        "  \"name\": \"object\",\n"
        "  \"visible\": true or false,\n"
        "  \"bbox\": [x1, y1, x2, y2] or null\n"
        "}\n\n"
        "If the object is NOT visible or does NOT exist in the image, set visible=false and bbox=null.\n\n"
        "Answer now.\nASSISTANT\n"
    )


def build_attribute_prompt(
    image_width: int, image_height: int, entity: Dict[str, Any], attribute: str
) -> str:
    return (
        "SYSTEM\nYou are a helpful assistant.\nUSER\n"
        "You are evaluating a generated image against a scene graph.\n"
        "Return ONLY valid JSON with no extra commentary.\n\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n" 
        f"Image size: width={image_width}, height={image_height}.\n\n"
        "Task: Determine whether the object has the specified attribute.\n"
        f"Object: id={entity.get('id')}, name={entity.get('name')}\n"
        f"Attribute: {attribute}\n\n"
        "Output JSON format:\n"
        "{\n"
        "  \"id\": 0,\n"
        "  \"attribute\": \"attr\",\n"
        "  \"satisfies\": \"yes|no|unclear\"\n"
        "}\n\n"
        "Answer now.\nASSISTANT\n"
    )


def build_relation_prompt(
    image_width: int, image_height: int, relation: Dict[str, Any], entities_by_id: Dict[Any, Dict[str, Any]]
) -> str:
    subj = entities_by_id.get(relation.get("subject"), {})
    obj = entities_by_id.get(relation.get("object"), {})
    return (
        "SYSTEM\nYou are a helpful assistant.\nUSER\n"
        "You are evaluating a generated image against a scene graph.\n"
        "Return ONLY valid JSON with no extra commentary.\n\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n" 
        f"Image size: width={image_width}, height={image_height}.\n\n"
        "Task: Determine whether the relationship is visible in the image.\n"
        f"Subject: id={relation.get('subject')}, name={subj.get('name')}\n"
        f"Relation: {relation.get('relation')}\n"
        f"Object: id={relation.get('object')}, name={obj.get('name')}\n\n"
        "Output JSON format:\n"
        "{\n"
        "  \"subject\": 0,\n"
        "  \"relation\": \"rel\",\n"
        "  \"object\": 1,\n"
        "  \"satisfies\": \"yes|no|unclear\"\n"
        "}\n\n"
        "Answer now.\nASSISTANT\n"
    )


def strip_reasoning(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def normalize_answer(value: Any) -> str:
    if value is None:
        return "unclear"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"yes", "no", "unclear"}:
            return cleaned
        if cleaned in {"true", "false"}:
            return "yes" if cleaned == "true" else "no"
    return "unclear"


def normalize_bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def normalize_visible(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "false"}:
            return cleaned == "true"
    return None


def image_path_from_pattern(
    pattern: str, images_dir: str, index: int, generation: int
) -> str:
    return os.path.join(images_dir, pattern.format(index=index, generation=generation))


def apply_batch_results(outputs: List[Any], pending_meta: List[Dict[str, Any]], results: Dict[int, Dict[str, Any]]) -> None:
    for meta, output in zip(pending_meta, outputs):
        raw_text = strip_reasoning(output.outputs[0].text)
        evaluation = extract_json(raw_text)
        image_result = meta["image_result"]
        p_result = results[image_result["prompt_index"]]
        if meta["task"] == "object":
            ent = meta["entity"]
            bbox = normalize_bbox(evaluation.get("bbox")) if evaluation else None
            visible = normalize_visible(evaluation.get("visible")) if evaluation else None
            image_result["evaluation"]["objects"].append(
                {
                    "id": ent.get("id"),
                    "name": ent.get("name"),
                    "visible": visible,
                    "bbox": bbox,
                }
            )
            # image_result["raw_responses"]["objects"].append(raw_text)
            gidx = image_result["generation_index"]
            p_result["label"][gidx] = "good" if p_result["label"][gidx] == "good" and visible else "bad"
        elif meta["task"] == "attribute":
            ent = meta["entity"]
            attribute = meta["attribute"]
            satisfies = normalize_answer(evaluation.get("satisfies") if evaluation else None)
            image_result["evaluation"]["attributes"].append(
                {"id": ent.get("id"), "attribute": attribute, "satisfies": satisfies}
            )
            # image_result["raw_responses"]["attributes"].append(raw_text)
            p_result["label"][gidx] = "good" if p_result["label"][gidx] == "good" and satisfies == "yes" else "bad"
        else:
            rel = meta["relation"]
            satisfies = normalize_answer(evaluation.get("satisfies") if evaluation else None)
            image_result["evaluation"]["relations"].append(
                {
                    "subject": rel.get("subject"),
                    "relation": rel.get("relation"),
                    "object": rel.get("object"),
                    "satisfies": satisfies,
                }
            )
            # image_result["raw_responses"]["relations"].append(raw_text)
            p_result["label"][gidx] = "good" if p_result["label"][gidx] == "good" and satisfies == "yes" else "bad"
        if evaluation is None:
            image_result["error"] = "parse_failed"


def summarize_results(results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np
    from modules.config import LABEL_THRESHOLD

    summary = {
        "prompts_total": len(results),
        "prompts_correct": 0,
        # "parse_failures": 0,
        "accuracy": None
    }

    for res in results.values():
        if (np.array(res["label"]) == "good").mean() > LABEL_THRESHOLD: 
            summary["prompts_correct"] += 1
            
    if len(results) > 0: summary["accuracy"] = summary["prompts_correct"] / summary["prompts_total"]
    return summary
