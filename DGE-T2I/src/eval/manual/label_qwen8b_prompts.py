"""
Manual labeling helper for qwen8b_t2i_prompts_aug_sample.json.

This script copies each image into src/eval/manual and prompts the user
to provide a label, saving labels to a JSONL file.
"""

import argparse
import json
import random
import os
import shutil
import sys
from typing import Dict, List, Optional, Set


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def resolve_index(entry: Dict) -> Optional[int]:
    index = entry.get("index")
    if isinstance(index, int):
        return index
    return None


def resolve_image_paths(index: int, images_dir: str) -> List[str]:
    paths = []
    for generation in range(1, 6):
        name = f"{index:04d}-{generation}.png"
        image_path = os.path.join(images_dir, name)
        if os.path.exists(image_path):
            paths.append(image_path)
    return paths


def load_existing_labels(label_path: str) -> Set[str]:
    labels = set()
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_source = record.get("image_source")
            if isinstance(image_source, str):
                labels.add(image_source)
    return labels


def main() -> int:
    root = project_root()
    default_output_dir = os.path.join(root, "src", "eval", "manual")
    default_images_root = os.path.join(root, "data", "images")
    default_label_file = os.path.join(default_output_dir, "labels.jsonl")

    parser = argparse.ArgumentParser(description="Manually label generated images.")
    parser.add_argument(
        "--input_json",
        type=str,
        default=os.path.join(root, "data", "raw", "qwen8b_t2i_prompts_aug_sample.json"),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Folder name under data/images (e.g., qwen-image, flux).",
    )
    parser.add_argument("--images_root", type=str, default=default_images_root)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--label_file", type=str, default=default_label_file)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Relabel entries already labeled.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError("Input JSON must be a list of examples.")

    images_dir = os.path.join(args.images_root, args.model)
    if not os.path.isdir(images_dir):
        raise ValueError(f"Model images directory not found: {images_dir}")

    existing_labels = set() if args.force else load_existing_labels(args.label_file)
    end_index = len(entries) if args.limit is None else min(len(entries), args.start + args.limit)
    items = []
    for idx in range(args.start, end_index):
        entry = entries[idx]
        prompt_index = resolve_index(entry)
        if prompt_index is None:
            print(f"[{idx:04d}] Missing index in entry; skipping.")
            continue
        image_paths = resolve_image_paths(prompt_index, images_dir)
        if len(image_paths) != 5:
            print(f"[{idx:04d}] Expected 5 images for index {prompt_index}, found {len(image_paths)}; skipping.")
            continue
        prompt = entry.get("prompt", "")
        for image_path in image_paths:
            if not args.force and image_path in existing_labels:
                continue
            items.append((prompt_index, prompt, image_path))

    random.shuffle(items)
    output_path = os.path.join(args.output_dir, "current_image.png")
    with open(args.label_file, "a", encoding="utf-8") as label_handle:
        for prompt_index, prompt, image_path in items:
            shutil.copy2(image_path, output_path)

            print("\n" + "=" * 80)
            print(f"[{prompt_index:04d}] Prompt: {prompt}")
            print(f"Image path: {output_path}")
            label = input("Label [good/bad/unsure] (or type 'skip'/'quit'): ").strip().lower()
            if label.lower() in {"quit", "q", "exit"}:
                print("Stopping labeling session.")
                break
            if not label or label == "skip":
                continue
            if label not in {"good", "bad", "unsure"}:
                print("Invalid label. Use: good, bad, unsure.")
                continue

            record = {
                "index": prompt_index,
                "prompt": prompt,
                "image_source": image_path,
                "image_output": output_path,
                "label": label,
                "model": args.model,
            }
            label_handle.write(json.dumps(record) + "\n")
            label_handle.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
