"""Module for processing scene graph data files."""

import json
from pathlib import Path
from typing import List, Dict, Any
import random
from tqdm import tqdm
import numpy as np
from .config import MAX_ITEMS_PER_SCENE, THRESHOLD, SCORE_THRESHOLD, GAMMA, BETA, PROMPTS_PER_SG, MAX_RELATIONS_PER_SCENE
import copy 

def generate_question(data: dict, obj_counts: dict) -> str:
    """Generates a T2I prompt for a given scene graph and the counts of objects used already.

    Args:
      data: target scene graph, formatted as 
          {'image_id', 
          'filename', 
          'width', 
          'height', 
          'entities': [list of {'id': [integer], 'name': [string], 'attributes': [list of strings], 'bbox': [list of floats], 'relative_area': [float]}], 
          'relations'[list of {'subject': [integer id], 'relation': [string], 'object': [id], 'score': [float]}]} 
      obj_counts: 

    Returns:
        Formatted prompt string for LLM processing
    """
    prompt = """Generate a concise caption for an image containing the following objects:\n"""
    prev_oc = copy.deepcopy(obj_counts)
    # Limit to MAX_ITEMS_PER_SCENE items and filter attributes
    curr = {"items": [], "relationships": []}
    items = data.get('entities', [])
    item_map = {}
    for x in items:
      if obj_counts.get(x['name'], 0) <= THRESHOLD and x['id'] not in item_map:
        item_map[x['id']] = x
    items = list(item_map.values())

    if not items: return None, obj_counts

    # Sample MAX_ITEMS_PER_SCENE objects, prioritizing bigger objects and part of OVAD. 
    areas = []
    non_ovad = []
    for i in range(len(items)):
      item = items[i] 
      if 'bbox' not in item: 
        non_ovad.append(i)
        areas.append(0)
      else: 
        areas.append(item['bbox'][2] * item['bbox'][3])

    areas = np.array(areas, dtype=np.float64)
    if len(areas) > len(non_ovad): # avoids sum by 0 
      areas = 0 if areas.sum() == 0 else areas/areas.sum() * (0.7 if len(non_ovad) > 0 else 1) 
    for idx in non_ovad: 
      areas[idx] = (0.3 if len(areas) > len(non_ovad) else 1) / len(non_ovad)
    
    items = np.random.choice(items, size=min(MAX_ITEMS_PER_SCENE, len(items)), replace=False, p=areas)
    ids = set(x['id'] for x in items)

    # add object list to prompt
    for item in items: 
      quantity = 1
      description = ""
      obj_counts[item['name']] = obj_counts.get(item['name'], 0) + 1
      
      prompt += f"   - {quantity} {item['name']} (object id : {item['id']})\n" 
      for attr in item['attributes']: 
        prompt +=  f"      -{attr}\n" 
    
    # add relationship list to prompt
    relationship_list = ""
    relations = data.get("relations", [])
    for relation in random.sample(relations, min(MAX_RELATIONS_PER_SCENE, len(relations))) : 
      if relation['subject'] in ids and relation['object'] in ids and relation['score'] >= SCORE_THRESHOLD: 
        relationship_list += f"   - Object {relation['subject']} {relation['relation']} object {relation['object']}\n" 
    
    if len(relationship_list) > 0: 
      prompt += f"And containing the following relationships:\n{relationship_list}"
    else: 
      return None, prev_oc 
    return prompt, obj_counts


def process_data(file_name: str, sample=None) -> List[Dict[str, str]]:
    """Process all or a subset of items from the given file and return a list of prompts."""
    with open(file_name, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    if sample is not None:
        data = random.sample(data, sample)
    
    prompts = []
    img_filenames = []
    obj_counts = {}
    questions = set()
    for img_data in tqdm(data, desc="Creating prompts for LLM"):
      for i in range(PROMPTS_PER_SG):
        try:
            question, obj_counts = generate_question(img_data, obj_counts)
            if not question or question in questions: continue
            questions.add(question)
            prompt = (
                "SYSTEM\nYou are a helpful assistant.\n"
                f"USER\n{question}\n"
                f"ASSISTANT\n"
            )
            inputs = {
                "prompt": prompt
            }
            prompts.append(inputs)
            img_filenames.append({"filename": img_data["filename"]})
        except Exception as e:
            print(f"Error processing {img_data['filename']}: {str(e)}")

    return prompts, img_filenames