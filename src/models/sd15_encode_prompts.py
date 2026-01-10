import argparse
import json
import os

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from tqdm import tqdm


DEFAULT_DATA_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_EMBEDDINGS_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings"


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


def normalize_embeddings(encoded):
    prompt_embeds = None
    pooled_prompt_embeds = None

    if isinstance(encoded, dict):
        prompt_embeds = encoded.get("prompt_embeds")
        pooled_prompt_embeds = encoded.get("pooled_prompt_embeds")
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    if isinstance(encoded, tuple):
        if len(encoded) >= 2:
            prompt_embeds = encoded[0]
            pooled_prompt_embeds = encoded[1]
        elif len(encoded) == 1:
            prompt_embeds = encoded[0]
        else:
            prompt_embeds = encoded[0]
            pooled_prompt_embeds = encoded[2] if len(encoded) > 2 else None
    else:
        prompt_embeds = encoded

    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def main():
    parser = argparse.ArgumentParser(description="Encode prompts to embeddings for T2I generation")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR, help="Directory to save embeddings")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON key containing the prompt text")
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism (e.g., balanced, auto, sequential, none)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts with existing embeddings")
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

    def move_text_encoders(pipeline):
        for name in ("text_encoder", "text_encoder_2"):
            module = getattr(pipeline, name, None)
            if module is not None:
                module.to(device)

    def load_pipeline():
        if device_map:
            return AutoPipelineForText2Image.from_pretrained(
                args.model_id,
                torch_dtype=dtype,
                device_map=device_map
            )
        pipeline = AutoPipelineForText2Image.from_pretrained(
            args.model_id,
            torch_dtype=dtype
        )
        move_text_encoders(pipeline)
        return pipeline

    try:
        pipeline = load_pipeline()
    except ValueError:
        if device_map:
            pipeline = DiffusionPipeline.from_pretrained(
                args.model_id,
                torch_dtype=dtype,
                device_map=device_map
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.model_id,
                torch_dtype=dtype
            )
            move_text_encoders(pipeline)
        if not hasattr(pipeline, "encode_prompt"):
            raise ValueError(
                "Loaded pipeline does not support encode_prompt; "
                "please provide a model with a text-to-image pipeline that exposes encode_prompt."
            )
    execution_device = getattr(pipeline, "_execution_device", device)

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
            encoded = pipeline.encode_prompt(
                prompt=prompt,
                device=execution_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
        embeddings = normalize_embeddings(encoded)
        def to_cpu(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return [to_cpu(v) for v in value]
            return value.detach().cpu()

        save_payload = {k: to_cpu(v) for k, v in embeddings.items()}
        torch.save(save_payload, out_path)

    print(f"Encoding complete. Embeddings saved to {args.embeddings_dir}")


if __name__ == "__main__":
    main()
