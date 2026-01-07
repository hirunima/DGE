"""Module for processing scene graph data files."""

import json
from pathlib import Path
from typing import List, Dict, Any
import random
from tqdm import tqdm
import numpy as np
from .config import *
import copy 

def augment_scene_graphs(data: List[Dict[str, Any]], max_aug_per_item: int = 1) -> List[Dict[str, Any]]:
    """Create augmented variants by mixing entities/relations across scene graphs."""
    augmented = []
    for scene in data:
        # Always keep the original scene, then add synthetic variants.
        augmented.append(scene)
        entities = scene.get("entities", [])
        relations = scene.get("relations", [])
        if not entities or len(data) < 2:
            continue

        other_scenes = [s for s in data if s is not scene and s.get("entities")]
        if not other_scenes:
            continue

        partner = random.choice(other_scenes)
        partner_entities = partner.get("entities", [])
        partner_relations = partner.get("relations", [])

        def _subset_entities(entity_list: List[Dict[str, Any]], max_keep: int) -> List[Dict[str, Any]]:
            keep = random.randint(1, min(len(entity_list), max_keep))
            return random.sample(entity_list, keep)

        def _filter_relations(rel_list: List[Dict[str, Any]], keep_ids: set, max_keep: int) -> List[Dict[str, Any]]:
            filtered = [
                r for r in rel_list
                if r.get("subject") in keep_ids and r.get("object") in keep_ids
            ]
            if len(filtered) > max_keep:
                filtered = random.sample(filtered, max_keep)
            return filtered

        def _reduce_attributes(entity_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            adjusted = []
            for ent in entity_list:
                ent_copy = copy.deepcopy(ent)
                attrs = ent_copy.get("attributes")
                if isinstance(attrs, list) and attrs:
                    # Reduce attributes to a smaller random subset for variation.
                    max_keep = min(len(attrs), 5)
                    keep = random.randint(1, max_keep)
                    ent_copy["attributes"] = random.sample(attrs, keep)
                adjusted.append(ent_copy)
            return adjusted

        def _remap_entities_relations(entity_list: List[Dict[str, Any]], rel_list: List[Dict[str, Any]]):
            id_map = {ent.get("id"): idx for idx, ent in enumerate(entity_list)}
            remapped_entities = []
            for idx, ent in enumerate(entity_list):
                ent_copy = copy.deepcopy(ent)
                ent_copy["id"] = idx
                remapped_entities.append(ent_copy)
            remapped_relations = []
            for rel in rel_list:
                subj = rel.get("subject")
                obj = rel.get("object")
                if subj in id_map and obj in id_map:
                    rel_copy = copy.deepcopy(rel)
                    rel_copy["subject"] = id_map[subj]
                    rel_copy["object"] = id_map[obj]
                    remapped_relations.append(rel_copy)
            return remapped_entities, remapped_relations

        candidates = []

        # Variant 1: combine entities/relations from both scenes.
        base_subset = _subset_entities(entities, max_keep=MAX_ITEMS_PER_SG)
        partner_subset = _subset_entities(partner_entities, max_keep=MAX_ITEMS_PER_SG)
        combined_entities = base_subset + partner_subset
        base_ids = {e.get("id") for e in base_subset}
        partner_ids = {e.get("id") for e in partner_subset}
        combined_relations = _filter_relations(relations, base_ids, MAX_RELATIONS_PER_SG)
        combined_relations += _filter_relations(partner_relations, partner_ids, MAX_RELATIONS_PER_SG)
        candidates.append(("combine", combined_entities, combined_relations))
        
        # Variant 2: swap attributes only (keep entities/relations from the base scene).
        attr_swap_count = min(SWAP_COUNT, len(entities), len(partner_entities))
        if attr_swap_count > 0:
            base_attr_swapped = copy.deepcopy(entities)
            base_indices = random.sample(range(len(base_attr_swapped)), attr_swap_count)
            partner_samples = random.sample(partner_entities, attr_swap_count)
            for idx, repl in zip(base_indices, partner_samples):
                if isinstance(repl.get("attributes"), list):
                    base_attr_swapped[idx]["attributes"] = repl.get("attributes", [])
            attr_ids = {e.get("id") for e in base_attr_swapped}
            attr_relations = _filter_relations(relations, attr_ids, MAX_RELATIONS_PER_SG)
            candidates.append(("swap_attributes", base_attr_swapped, attr_relations))

        # Variant 3: swap relationship names only (keep subjects/objects from the base scene).
        if relations and partner_relations:
            relation_names = [r.get("relation") for r in partner_relations if r.get("relation")]
            if relation_names:
                rel_name_swapped = copy.deepcopy(relations)
                for rel in rel_name_swapped:
                    rel["relation"] = random.choice(relation_names)
                rel_name_ids = {e.get("id") for e in entities}
                rel_name_relations = _filter_relations(rel_name_swapped, rel_name_ids, MAX_RELATIONS_PER_SG)
                candidates.append(("swap_relation_names", list(entities), rel_name_relations))

        random.shuffle(candidates)
        for aug_idx, (aug_type, aug_entities, aug_relations) in enumerate(candidates[:max_aug_per_item]):
            # Remap IDs to keep them unique and compact per augmented scene.
            reduced_entities = _reduce_attributes(aug_entities)
            remapped_entities, remapped_relations = _remap_entities_relations(reduced_entities, aug_relations)
            new_scene = copy.deepcopy(scene)
            new_scene["entities"] = remapped_entities
            new_scene["relations"] = remapped_relations
            new_scene["augmented"] = True
            new_scene["augmented_from"] = [
                scene.get("filename", scene.get("image_id")),
                partner.get("filename", partner.get("image_id")),
            ]
            new_scene["augmentation_type"] = aug_type
            new_scene["augmentation_index"] = aug_idx
            augmented.append(new_scene)

    return augmented

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
    valid_ids = set()
    valid_items = []
    for x in items:
        if obj_counts.get(x['name'], 0) <= THRESHOLD and x['id'] not in valid_ids:
            valid_ids.add(x['id'])
            valid_items.append(x)
    items = valid_items

    if not items: 
        return None, obj_counts

    ids = set(x['id'] for x in items)
    relationships_str = ""
    relations = data.get("relations", [])
    
    valid_relations = [r for r in relations if r['subject'] in ids and r['object'] in ids and r['score'] >= SCORE_THRESHOLD]
    
    if not valid_relations:
        return None, prev_oc

    selected_relations = random.sample(valid_relations, min(MAX_RELATIONS_PER_SCENE, len(valid_relations)))
   
    sel_ids = set()
    for relation in selected_relations:
        relationships_str += f"   - Object {relation['subject']} {relation['relation']} Object {relation['object']}\n" 
        sel_ids.add(relation['subject'])
        sel_ids.add(relation['object'])
    
    if not relationships_str: 
        return None, prev_oc 
   
    forced_items = [x for x in items if x['id'] in sel_ids]
    pool_items = [x for x in items if x['id'] not in sel_ids]

    areas = []
    non_ovad = []
    
    for i in range(len(pool_items)):
        item = pool_items[i] 
        if 'bbox' not in item: 
            non_ovad.append(i)
            areas.append(0)
        else: 
            areas.append(np.log1p(item['bbox'][2] * item['bbox'][3]))

    areas = np.array(areas, dtype=np.float64)
    if len(areas) > len(non_ovad): 
        areas = np.zeros(areas.shape) if areas.sum() == 0 else areas/areas.sum() * (OVAD_P if len(non_ovad) > 0 else 1) 
    
    for idx in non_ovad: 
        areas[idx] = ((1 - OVAD_P) if len(areas) > len(non_ovad) else 1) / len(non_ovad)

    slots_remaining = max(0, MAX_ITEMS_PER_SCENE - len(sel_ids))
    sample_size = min(slots_remaining, len(pool_items))
    
    if sample_size > 0:
        chosen_pool = np.random.choice(pool_items, size=sample_size, replace=False, p=areas)
        items = np.concatenate([forced_items, chosen_pool])
    else:
        items = np.array(forced_items)
        
    objects_str = ""
    for item in items: 
        obj_counts[item['name']] = obj_counts.get(item['name'], 0) + 1
        
        objects_str += f"   - 1 {item['name']} (object id : {item['id']})\n" 
        for attr in item['attributes']: 
            objects_str +=  f"      -{attr}\n" 
    

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

   # TODO: augment existing data, creating new scene graphs
    data = augment_scene_graphs(data)

    if sample is not None:
        data = random.sample(data, sample)
    
    prompts = []
    img_filenames = []
    obj_counts = {}
    questions = set()
    counts =[]
    for img_data in tqdm(data, desc="Creating prompts for LLM"):
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
