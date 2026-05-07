"""Main evaluation pipeline workflow."""

import json
import os
from typing import Any, Dict, List

from PIL import Image
from torch.cuda import is_available as cuda_available

from tqdm import tqdm

from .processing import (
    apply_batch_results,
    build_attribute_prompt,
    build_object_prompt,
    build_relation_prompt,
    image_path_from_pattern,
    load_json_or_jsonl,
    summarize_results,
    extract_scene_graph
)

from modules.config import PROJECT_ROOT


def main_pipeline(args: Any, model: Any, sampling_params: Any) -> None:
    if not cuda_available():
        raise RuntimeError(
            "CUDA is not available. Qwen3-VL via vLLM requires a CUDA-capable GPU. "
            "Run on a GPU node or enable CUDA before launching this script."
        )

    sampling_params.max_tokens = args.max_tokens

    if not args.sg_file: 
        try:
            prompts_data = load_json_or_jsonl(args.prompts_file)
        except Exception as e:
            raise Exception("Got no scene graph file and couldn't load prompts file.") 
    else: 
        sgs = load_json_or_jsonl(args.sg_file)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{args.model}_eval_results.json") 
    results: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(results_path): 
        with open(results_path, "r") as f:
            results = json.load(f)

    start_idx = args.start_idx
    if args.prompts_file:
        end_idx = args.end_idx if args.end_idx is not None else len(prompts_data)
    else:
        end_idx = args.end_idx if args.end_idx is not None else len(sgs)
    total = end_idx - start_idx
    if args.limit is not None:
        total = min(total, args.limit)
    end_idx = start_idx + total

    pending_prompts = []
    pending_meta = []

    images_dir = os.path.join(args.images_dir, args.model)

    for idx in tqdm(range(start_idx, end_idx)):
        if args.prompts_file: 
            entry = prompts_data[idx]
            scene_graph = extract_scene_graph(entry["meta_prompt"]["prompt"]) if "meta_prompt" in entry else None
        else: 
            scene_graph = sgs[idx]
        if str(idx) in results or scene_graph is None or "error" in scene_graph:
            continue

        results[idx] = {"data": [], "label": ["good" for _ in range(args.generation)]}

        for i in range(args.generation):
            if args.prompts_file:
                image_path = image_path_from_pattern(args.image_pattern, images_dir, idx, i + 1)
            else: 
                image_path = os.path.join(images_dir, scene_graph["filename"])
            if not os.path.exists(image_path):
                print("Warning: didn't find image at path ", image_path)
                continue

            with Image.open(image_path) as img:
                image = img.convert("RGB").copy()
            width, height = image.size

            entities, relations = scene_graph["objects"], scene_graph["relations"]
            entities_by_id = {ent.get("id"): ent for ent in entities}

            image_result: Dict[str, Any] = {
                "prompt_index": idx,
                "generation_index": i, 
                "image_path": image_path,
                "prompt": entry.get("prompt") if args.prompts_file else scene_graph["filename"].split("_", 2)[2].rsplit(".", 1)[0],
                # "scene_graph": {"entities": entities, "relations": relations},
                "evaluation": {"objects": [], "attributes": [], "relations": []},
                # "raw_responses": {"objects": [], "attributes": [], "relations": []},
                "error": None,
            }

            for ent in entities:
                prompt_text = build_object_prompt(width, height, ent)
                pending_prompts.append(
                    {
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image},
                    }
                )
                pending_meta.append(
                    {
                        "image_result": image_result,
                        "task": "object",
                        "entity": ent,
                    }
                )

            for ent in entities:
                for attribute in ent.get("attributes") or []:
                    prompt_text = build_attribute_prompt(width, height, ent, attribute)
                    pending_prompts.append(
                        {
                            "prompt": prompt_text,
                            "multi_modal_data": {"image": image},
                        }
                    )
                    pending_meta.append(
                        {
                            "image_result": image_result,
                            "task": "attribute",
                            "entity": ent,
                            "attribute": attribute,
                        }
                    )

            for rel in relations:
                prompt_text = build_relation_prompt(width, height, rel, entities_by_id)
                pending_prompts.append(
                    {
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image},
                    }
                )
                pending_meta.append(
                    {
                        "image_result": image_result,
                        "task": "relation",
                        "relation": rel,
                    }
                )

            results[idx]["data"].append(image_result)

        if args.batch_size and len(pending_prompts) >= args.batch_size:
            outputs = model.generate(pending_prompts, sampling_params=sampling_params, use_tqdm=False)
            apply_batch_results(outputs, pending_meta, results)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            pending_prompts = []
            pending_meta = []

    if pending_prompts:
        outputs = model.generate(pending_prompts, sampling_params=sampling_params, use_tqdm=True)
        apply_batch_results(outputs, pending_meta, results)

    summary = summarize_results(results)
    summary_path = os.path.join(args.output_dir, f"{args.model}_eval_summary.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {results_path}")
    print(f"Saved summary to {summary_path}")
