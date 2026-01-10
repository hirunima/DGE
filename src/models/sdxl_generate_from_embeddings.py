import argparse
import json
import os

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm


DEFAULT_DATA_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_EMBEDDINGS_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings"
DEFAULT_IMAGES_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images"
DEFAULT_SEED = 44


def check_images_exist(example_id, output_dir, num_generations):
    for i in range(1, num_generations + 1):
        image_path = os.path.join(output_dir, f"{example_id}-{i}.png")
        if not os.path.exists(image_path):
            return False
    return True


def resolve_indices(data_len, start_idx, end_idx, num_splits, split_id):
    if start_idx is not None and end_idx is not None:
        return start_idx, end_idx

    if split_id >= num_splits:
        raise ValueError(f"split_id ({split_id}) must be less than num_splits ({num_splits})")
    if split_id < 0:
        raise ValueError("split_id must be non-negative")

    items_per_split = data_len // num_splits
    remainder = data_len % num_splits

    if split_id < remainder:
        start_idx = split_id * (items_per_split + 1)
        end_idx = start_idx + items_per_split + 1
    else:
        start_idx = split_id * items_per_split + remainder
        end_idx = start_idx + items_per_split

    return start_idx, end_idx


def main():
    parser = argparse.ArgumentParser(description="Generate images from pre-encoded embeddings")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR, help="Directory containing embeddings")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR, help="Directory to save generated images")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of images per prompt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    # parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of diffusion steps")
    # parser.add_argument("--high_noise_frac", type=float, default=0.8, help="Fraction of steps to run in base model")
    # parser.add_argument("--refiner_model_id", type=str, default="", help="Optional SDXL refiner model ID or local path")
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism (e.g., balanced, auto, sequential, none)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts that already have images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_id}")
    device_map = args.device_map
    if device_map == "none":
        device_map = None
    if torch.cuda.device_count() < 2:
        device_map = None


    base = DiffusionPipeline.from_pretrained(args.model_id, 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16",
                device_map=device_map
                )

    if not device_map:
        base = base.to(device)

    base.enable_vae_slicing()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected data to be a list of prompt items")

    start_idx, end_idx = resolve_indices(len(data), args.start_idx, args.end_idx, args.num_splits, args.split_id)
    print(f"Generating images for prompts {start_idx} to {end_idx - 1}")

    os.makedirs(args.images_dir, exist_ok=True)

    skipped_count = 0
    generated_count = 0

    for idx in tqdm(range(start_idx, end_idx)):
        example_id = f"{idx:04d}"
        if args.skip_existing and check_images_exist(example_id, args.images_dir, args.num_generations):
            skipped_count += 1
            continue

        embedding_path = os.path.join(args.embeddings_dir, f"{example_id}.pt")
        if not os.path.exists(embedding_path):
            continue

        payload = torch.load(embedding_path, map_location=device)

        for e in payload: 
            payload[e] = payload[e].to(device) if payload[e] != None else None

        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        call_kwargs = {
            **payload, 
            "num_images_per_prompt": args.num_generations,
            "generator": generator
        }

        images = base(**call_kwargs).images

        for i, img in enumerate(images):
            img.save(os.path.join(args.images_dir, f"{example_id}-{i+1}.png"))

        generated_count += 1

    print(f"Generation complete. Images saved to {args.images_dir}")
    if args.skip_existing:
        print(f"Skipped: {skipped_count} prompts (already generated)")
    print(f"Generated: {generated_count} prompts")


if __name__ == "__main__":
    main()
