#!/usr/bin/env python3
"""Run ablation permutations sequentially to avoid OOM."""

import subprocess
import sys
import json
from pathlib import Path

# All 8 permutations
permutations = [
    ("E1", "E2", "E3"),
    ("E1", "E2", "V3"),
    ("E1", "V2", "E3"),
    ("E1", "V2", "V3"),
    ("V1", "E2", "E3"),
    ("V1", "E2", "V3"),
    ("V1", "V2", "E3"),
    ("V1", "V2", "V3"),
]

# Build backend specs for each permutation
def build_cmd(perm):
    cmd = [
        sys.executable, "src/eval/ablation.py",
        "--output-dir", "./reports/ablation",
        "--images-dir", "./data/images/split/train/qwen-image",
        "--prompts-file", "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json",
        "--low-vram",
    ]
    # Stage 1 - only set the backend kind for the one being used
    if perm[0] == "E1":
        cmd.extend(["--e1-backend-kind", "grounding-dino", "--eupe-model-path", "IDEA-Research/grounding-dino-base"])
    else:
        cmd.extend(["--v1-backend-kind", "qwen", "--qwen-model-path", "/fs/nexus-projects/scene_graph_sd/Qwen3-VL-8B-Instruct"])
    # Stage 2
    if perm[1] == "E2":
        cmd.extend(["--e2-backend-kind", "siglip", "--siglip-model-path", "google/siglip2-so400m-patch14-384"])
    else:
        cmd.extend(["--v2-backend-kind", "qwen", "--qwen-model-path", "/fs/nexus-projects/scene_graph_sd/Qwen3-VL-8B-Instruct"])
    # Stage 3
    if perm[2] == "E3":
        cmd.extend(["--e3-backend-kind", "siglip", "--siglip-model-path", "google/siglip2-so400m-patch14-384"])
    else:
        cmd.extend(["--v3-backend-kind", "qwen", "--qwen-model-path", "/fs/nexus-projects/scene_graph_sd/Qwen3-VL-8B-Instruct"])
    # Only run this specific permutation
    cmd.extend(["--backends", f"{perm[0]},{perm[1]},{perm[2]}"])
    return cmd

def check_completed(permutations):
    completed = []
    for perm in permutations:
        filename = f"./reports/ablation/permutations/{perm[0]}-{perm[1]}-{perm[2]}_details.json"
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            if size > 10:  # More than just "{}"
                completed.append(perm)
    return completed

def main():
    # Check what's already completed
    completed = check_completed(permutations)
    print(f"Already completed: {[f'{p[0]}-{p[1]}-{p[2]}' for p in completed]}")

    # Run remaining permutations
    for perm in permutations:
        name = f"{perm[0]}-{perm[1]}-{perm[2]}"
        if perm in completed:
            print(f"Skipping {name} (already completed)")
            continue

        print(f"\n{'='*50}")
        print(f"Running permutation: {name}")
        print(f"{'='*50}")

        cmd = build_cmd(perm)
        env = {**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": "0"}

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"ERROR: Permutation {name} failed with code {result.returncode}")
            sys.exit(1)

        print(f"Completed: {name}")

    print("\nAll permutations completed!")

    # Show summary
    print("\nResults:")
    for perm in permutations:
        filename = f"./reports/ablation/permutations/{perm[0]}-{perm[1]}-{perm[2]}_details.json"
        path = Path(filename)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {perm[0]}-{perm[1]}-{perm[2]}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()

def check_completed(permutations):
    completed = []
    for perm in permutations:
        filename = f"./reports/ablation/permutations/{perm[0]}-{perm[1]}-{perm[2]}_details.json"
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            if size > 10:  # More than just "{}"
                completed.append(perm)
    return completed

def main():
    # Check what's already completed
    completed = check_completed(permutations)
    print(f"Already completed: {[f'{p[0]}-{p[1]}-{p[2]}' for p in completed]}")

    # Run remaining permutations
    for perm in permutations:
        name = f"{perm[0]}-{perm[1]}-{perm[2]}"
        if perm in completed:
            print(f"Skipping {name} (already completed)")
            continue

        print(f"\n{'='*50}")
        print(f"Running permutation: {name}")
        print(f"{'='*50}")

        cmd = build_cmd(perm)
        env = {**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": "0"}

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"ERROR: Permutation {name} failed with code {result.returncode}")
            sys.exit(1)

        print(f"Completed: {name}")

    print("\nAll permutations completed!")

    # Show summary
    print("\nResults:")
    for perm in permutations:
        filename = f"./reports/ablation/permutations/{perm[0]}-{perm[1]}-{perm[2]}_details.json"
        path = Path(filename)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {perm[0]}-{perm[1]}-{perm[2]}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
