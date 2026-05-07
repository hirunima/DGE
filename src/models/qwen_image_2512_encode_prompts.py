import argparse
import json
import os

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm


DEFAULT_DATA_PATH = "data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_EMBEDDINGS_DIR = "data/embeddings"


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


def unpack_prompt_embeddings(encoded):
    if isinstance(encoded, dict):
        return encoded.get("prompt_embeds"), encoded.get("pooled_prompt_embeds")
    if isinstance(encoded, tuple):
        if len(encoded) >= 4:
            return encoded[0], encoded[2]
        if len(encoded) == 2:
            return encoded[0], encoded[1]
        if len(encoded) == 1:
            return encoded[0], None
    return encoded, None


def main():
    parser = argparse.ArgumentParser(description="Encode prompts to embeddings for T2I generation")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR, help="Directory to save embeddings")
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism (e.g., balanced, auto, sequential, none)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts with existing embeddings")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    

    print(f"Loading model: {args.model_id}")
    device_map = args.device_map
    if device_map == "none":
        device_map = None
    if torch.cuda.device_count() < 2:
        device_map = None


    pipeline = DiffusionPipeline.from_pretrained(args.model_id, 
                scheduler=None, 
                vae=None, 
                transformer=None,
                torch_dtype=torch.bfloat16,
                use_safetensors=True, 
                device_map=device_map
                )
    

    if not device_map:
        pipeline.text_encoder.to(device)

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected data to be a list of prompt items")

    start_idx, end_idx = resolve_indices(len(data), args.start_idx, args.end_idx, args.num_splits, args.split_id)
    print(f"Encoding prompts {start_idx} to {end_idx - 1}")

    os.makedirs(args.embeddings_dir, exist_ok=True)

    for idx in tqdm(range(start_idx, end_idx)):
        item = data[idx]
        prompt = item.get("prompt")
        if not prompt:
            continue

        example_id = f"{idx:04d}"
        out_path = os.path.join(args.embeddings_dir, f"{example_id}.pt")
        if args.skip_existing and os.path.exists(out_path):
            continue

        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1
            )

        def to_cpu(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return [to_cpu(v) for v in value]
            return value.detach().cpu()

        save_payload = {
            "prompt_embeds": to_cpu(prompt_embeds),
            "prompt_embeds_mask":to_cpu(prompt_embeds_mask)
        }
        torch.save(save_payload, out_path)

    print(f"Encoding complete. Embeddings saved to {args.embeddings_dir}")


if __name__ == "__main__":
    main()
