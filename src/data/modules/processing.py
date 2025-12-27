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
    """Generates a T2I prompt for a given scene graph using In-Context Learning (ICL) and CoT.

    Args:
      data: target scene graph dict.
      obj_counts: dictionary tracking usage of objects.

    Returns:
        Formatted prompt string for LLM processing or None if criteria not met.
    """
    
    # 1. Define the static In-Context Learning Prompt Template
    # This includes the Task, Instructions, and the two fixed Examples.
    PROMPT_HEADER = """Task: Generate a concise caption for an image based on the provided structured data (objects, attributes, and relationships).

Instructions:
1. Object Definition: Create a descriptive noun phrase for each object ID using its attributes.
2. Relationship Mapping: Replace object IDs in the relationship list with the descriptive phrases from Step 1.
3. Inferred Relationships: For any object NOT listed in the relationships, create a logical spatial connection or role for it (e.g., "in the background," "nearby," "next to").
4. Synthesis: Combine the mapped and inferred relationships into a single, natural-sounding English sentence.

---
Example 1:
[Input Data]
Objects:
   - 1 person (object id : 0)
      -clothes color:blue
      -gender:male
      -action:running
   - 1 dog (object id : 1)
      -color:golden
      -breed:retriever
   - 1 tree (object id : 2)
      -type:oak
Relationships:
   - Object 0 chasing Object 1

[Step-by-Step Reasoning]
1. Object Definition:
   - Object 0: A male in blue clothes.
   - Object 1: A golden retriever.
   - Object 2: An oak tree.
2. Relationship Mapping:
   - [A male in blue clothes] chasing [A golden retriever].
3. Inferred Relationships:
   - Object 2 (oak tree) has no defined relationship. It is likely part of the environment. -> "past an oak tree."
4. Synthesis:
   - A male in blue clothes chases a golden retriever past an oak tree.

[Final Caption]
A male in blue clothes chases a golden retriever past an oak tree.

---
Example 2:
[Input Data]
Objects:
   - 1 person (object id : 5)
      -gender:woman
      -expression:smiling
      -clothes:red coat
   - 1 umbrella (object id : 6)
      -color:black
      -state:open
   - 1 car (object id : 7)
      -color:silver
Relationships:
   - Object 5 holding Object 6

[Step-by-Step Reasoning]
1. Object Definition:
   - Object 5: A smiling woman in a red coat.
   - Object 6: An open black umbrella.
   - Object 7: A silver car.
2. Relationship Mapping:
   - [A smiling woman in a red coat] holding [an open black umbrella].
3. Inferred Relationships:
   - Object 7 (silver car) is unconnected. Since this is likely an outdoor street scene, it is probably nearby. -> "standing near a silver car."
4. Synthesis:
   - A smiling woman in a red coat holds an open black umbrella while standing near a silver car.

[Final Caption]
A smiling woman in a red coat holds an open black umbrella while standing near a silver car.

---
Example 3:
[Input Data]
Objects:
   - 1 person (object id : 20)
      -action:writing
   - 1 notebook (object id : 21)
      -state:open
   - 1 pen (object id : 22)
      -color:blue
Relationships:
   - Object 20 writing in Object 21

[Step-by-Step Reasoning]
1. Object Definition:
   - Object 20: A person.
   - Object 21: An open notebook.
   - Object 22: A blue pen.
2. Relationship Mapping:
   - [A person] writing in [an open notebook].
3. Inferred Relationships:
   - Object 22 (blue pen) is unconnected. Since the person is writing, they are likely using the pen. -> "using a blue pen."
4. Synthesis:
   - A person is writing in an open notebook using a blue pen.

[Final Caption]
A person is writing in an open notebook using a blue pen.

---
Example 4:
[Input Data]
Objects:
   - 1 sandwich (object id : 30)
      -type:club
   - 1 plate (object id : 31)
      -color:white
   - 1 juice (object id : 32)
      -flavor:orange
Relationships:
   - Object 30 on top of Object 31

[Step-by-Step Reasoning]
1. Object Definition:
   - Object 30: A club sandwich.
   - Object 31: A white plate.
   - Object 32: Orange juice.
2. Relationship Mapping:
   - [A club sandwich] on top of [A white plate].
3. Inferred Relationships:
   - Object 32 (orange juice) is unconnected. It is likely in a glass next to the plate. -> "served with a glass of orange juice."
4. Synthesis:
   - A club sandwich sits on a white plate, served with a glass of orange juice.

[Final Caption]
A club sandwich sits on a white plate, served with a glass of orange juice.

---
Example 5:
[Input Data]
Objects:
   - 1 cup (object id : 40)
   - 1 table (object id : 41)
   - 1 chair (object id : 42)
Relationships:
   - Object 40 on Object 41

[Step-by-Step Reasoning]
1. Object Definition:
   - Object 40: A cup.
   - Object 41: A table.
   - Object 42: A chair.
2. Relationship Mapping:
   - [A cup] on [a table].
3. Inferred Relationships:
   - Object 42 (chair) is unconnected. It is likely placed next to the table. -> "near a chair."
4. Synthesis:
   - A cup is on a table near a chair.

[Final Caption]
A cup is on a table near a chair.

---
Current Task:
[Input Data]
Objects:
"""
    prev_oc = copy.deepcopy(obj_counts)
    
    items = data.get('entities', [])
    item_map = {}
    for x in items:
        if obj_counts.get(x['name'], 0) <= THRESHOLD and x['id'] not in item_map:
            item_map[x['id']] = x
    items = list(item_map.values())

    if not items: 
        return None, obj_counts

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
    if len(areas) > len(non_ovad): 
        areas = 0 if areas.sum() == 0 else areas/areas.sum() * (0.7 if len(non_ovad) > 0 else 1) 
    for idx in non_ovad: 
        areas[idx] = (0.3 if len(areas) > len(non_ovad) else 1) / len(non_ovad)
    
    items = np.random.choice(items, size=min(MAX_ITEMS_PER_SCENE, len(items)), replace=False, p=areas)
    ids = set(x['id'] for x in items)

    objects_str = ""
    for item in items: 
        obj_counts[item['name']] = obj_counts.get(item['name'], 0) + 1
        
        objects_str += f"   - 1 {item['name']} (object id : {item['id']})\n" 
        for attr in item['attributes']: 
            objects_str +=  f"      -{attr}\n" 
    
    relationships_str = ""
    relations = data.get("relations", [])
    
    valid_relations = [r for r in relations if r['subject'] in ids and r['object'] in ids and r['score'] >= SCORE_THRESHOLD]
    
    if not valid_relations:
        return None, prev_oc

    selected_relations = random.sample(valid_relations, min(MAX_RELATIONS_PER_SCENE, len(valid_relations)))

    for relation in selected_relations:
        relationships_str += f"   - Object {relation['subject']} {relation['relation']} Object {relation['object']}\n" 
    
    if not relationships_str: 
        return None, prev_oc 

    final_prompt = (
        f"{PROMPT_HEADER}"
        f"{objects_str}"
        f"Relationships:\n"
        f"{relationships_str}\n"
        f"[Step-by-Step Reasoning]"
    )
    
    return final_prompt, obj_counts


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