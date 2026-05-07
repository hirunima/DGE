import argparse

import json
import os

import torch
import gc
from diffusers import QwenImagePipeline

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory
from tqdm import tqdm


DEFAULT_DATA_PATH = "data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
DEFAULT_IMAGES_DIR = "data/images"
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


def repeat_embeds(embeds, count):
    if embeds is None or count <= 1:
        return embeds
    if embeds.shape[0] == count:
        return embeds
    return embeds.repeat_interleave(count, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate images from pre-encoded embeddings")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR, help="Directory containing embeddings")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR, help="Directory to save generated images")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of images per prompt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism (e.g., balanced, auto, sequential, none)")
    parser.add_argument("--keep-text-encoders", action="store_true", help="Keep text encoders loaded (default: skip)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts that already have images")
    args = parser.parse_args()

    rank = 128  # you can also use rank=128 model to improve the quality
    precision = "int4"

    # Load the model
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        "data/models/nunchaku-qwen-image/svdq-int4_r128-qwen-image.safetensors"
    )

    pipe = QwenImagePipeline.from_pretrained(args.model_id, transformer=transformer, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    if get_gpu_memory() > 18:
        pipe.enable_model_cpu_offload()
    else:
        print("enabling per layer offloading")
        # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
        transformer.set_offload(
            True, use_pin_memory=False, num_blocks_on_gpu=1
        )  # increase num_blocks_on_gpu if you have more VRAM
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()

    negative_prompt = " "  # using an empty string if you do not have specific concept to remove
    
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
        item = data[idx]
        prompt = item.get("prompt")
        if not prompt:
            continue

        example_id = f"{idx:04d}"
        if args.skip_existing and os.path.exists(os.path.join(args.images_dir, f"{example_id}-{args.num_generations}.png")):
            skipped_count += 1
            continue
        
        for i in range(args.num_generations):
            img = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=1664,
                height=928,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                num_images_per_prompt=1
            ).images[0]

            img.save(os.path.join(args.images_dir, f"{example_id}-{i+1}.png"))
            generated_count += 1

            # Optional: Force memory cleanup after each image
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Generation complete. Images saved to {args.images_dir}")
    if args.skip_existing:
        print(f"Skipped: {skipped_count} prompts (already generated)")
    print(f"Generated: {generated_count} prompts")


if __name__ == "__main__":
    main()
