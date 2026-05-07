import argparse
import json
import os

import torch
from diffusers import ZImagePipeline
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

def main():
    parser = argparse.ArgumentParser(description="Encode prompts to embeddings for Z-Image")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR, help="Directory to save embeddings")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON key containing the prompt text")
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts with existing embeddings")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    device_map = args.device_map
    if device_map == "none":
        device_map = None
    if torch.cuda.device_count() < 2:
        device_map = None

    pipe = ZImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        device_map=device_map
    )

    pipe.transformer.set_attention_backend("flash")    
    # pipe.transformer.compile()
    pipe.enable_model_cpu_offload()

    if not device_map:
        pipe.to(device)
        pipe.text_encoder.to(device)
        
    with open(args.data_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected data to be a list of prompt items")

    start_idx, end_idx = resolve_indices(len(data), args.start_idx, args.end_idx, args.num_splits, args.split_id)
    print(f"Encoding prompts {start_idx} to {end_idx - 1}")

    os.makedirs(args.embeddings_dir, exist_ok=True)

    for idx in tqdm(range(start_idx, end_idx)):
        item = data[idx]
        prompt = item.get(args.prompt_key)
        if not prompt:
            continue

        example_id = f"{idx:04d}"
        out_path = os.path.join(args.embeddings_dir, f"{example_id}.pt")
        if args.skip_existing and os.path.exists(out_path):
            continue

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                do_classifier_free_guidance=True,
            )
        
        def to_cpu(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return [to_cpu(v) for v in value]
            return value.detach().cpu()

        save_payload = {
            "prompt_embeds": to_cpu(prompt_embeds),
            "negative_prompt_embeds":to_cpu(negative_prompt_embeds)
        }

        torch.save(save_payload, out_path)

    print(f"Encoding complete. Embeddings saved to {args.embeddings_dir}")


if __name__ == "__main__":
    main()
