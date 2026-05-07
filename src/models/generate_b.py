import argparse
import json
import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DEFAULT_DATA_PATH = "data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_SEED = 44

def check_images_exist(example_id, output_dir, num_generations):
    # Quick check without loop for speed if possible
    return all(os.path.exists(os.path.join(output_dir, f"{example_id}-{i}.png")) 
               for i in range(1, num_generations + 1))

def resolve_indices(data_len, start_idx, end_idx, num_splits, split_id):
    if start_idx is not None and end_idx is not None:
        return start_idx, end_idx
    if split_id >= num_splits or split_id < 0:
        raise ValueError(f"Invalid split_id {split_id} for num_splits {num_splits}")

    items_per_split = data_len // num_splits
    remainder = data_len % num_splits

    if split_id < remainder:
        start_idx = split_id * (items_per_split + 1)
        end_idx = start_idx + items_per_split + 1
    else:
        start_idx = split_id * items_per_split + remainder
        end_idx = start_idx + items_per_split
    return start_idx, end_idx

def save_images_async(images, output_dir, example_id):
    """Helper to save images in a separate thread"""
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"{example_id}-{i+1}.png"))

def main():
    parser = argparse.ArgumentParser(description="Generate images from pre-encoded embeddings")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device-map", type=str, default=None) # Changed default to None
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--offload_strategy",
        type=str,
        default="group",
        choices=("group", "model", "sequential", "none"),
        help="Offload strategy to reduce VRAM usage without quantization",
    )
    args = parser.parse_args()

    # 1. Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Optimized Model Loading
    print(f"Loading model: {args.model_id}")
    
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "use_safetensors": True,
    }

    base = DiffusionPipeline.from_pretrained(args.model_id, **load_kwargs)
    base.transformer.set_attention_backend("flash")
    
    # Clean up pipeline
    base.text_encoder = None
    base.text_encoder_2 = None # Ensure secondary encoder is also gone if present
    base.tokenizer = None
    base.tokenizer_2 = None

    # 3. CRITICAL: Memory Management Strategy
    if device == "cpu":
        base = base.to(device)
    elif args.offload_strategy == "none":
        base = base.to(device)
    elif args.offload_strategy == "sequential":
        base.enable_sequential_cpu_offload()
    elif args.offload_strategy == "model":
        base.enable_model_cpu_offload()
    else:
        if hasattr(base, "enable_group_offload"):
            base.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
            )
        else:
            base.enable_model_cpu_offload()

    # 4. Compilation (Major speedup for Flux/SD3)
    # mode="reduce-overhead" is faster to compile, "max-autotune" is faster to run
    print("Compiling transformer... (this takes a minute but speeds up generation)")
    base.transformer = torch.compile(base.transformer, mode="max-autotune")

    # Disable VAE slicing unless necessary (it slows down generation significantly)
    base.enable_vae_slicing() 
    base.enable_vae_tiling()

    # Data Setup
    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx, end_idx = resolve_indices(len(data), args.start_idx, args.end_idx, args.num_splits, args.split_id)
    print(f"Generating images for prompts {start_idx} to {end_idx - 1}")

    os.makedirs(args.images_dir, exist_ok=True)
    skipped_count = 0
    generated_count = 0

    # Thread pool for saving images
    executor = ThreadPoolExecutor(max_workers=4)

    # 5. Pipeline Configuration
    # If the model allows, reduce steps. Default is usually 28-50. 
    # If this is Flux-Schnell or SDXL-Lightning, set num_inference_steps=4
    # inference_kwargs = {"num_inference_steps": 28} # Uncomment to enforce lower steps

    for idx in tqdm(range(start_idx, end_idx)):
        example_id = f"{idx:04d}"
        
        # Fast skip check
        if args.skip_existing and check_images_exist(example_id, args.images_dir, args.num_generations):
            skipped_count += 1
            continue

        embedding_path = os.path.join(args.embeddings_dir, f"{example_id}.pt")
        if not os.path.exists(embedding_path):
            continue

        # Load embedding directly to GPU to avoid CPU-GPU transfer lag
        payload = torch.load(embedding_path, map_location=device, weights_only=True)
        
        # Ensure correct dtype
        for k, v in payload.items():
            if torch.is_tensor(v) and v.dtype == torch.float32:
                payload[k] = v.to(dtype=torch.bfloat16)

        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        
        call_kwargs = {
            **payload,
            "num_images_per_prompt": args.num_generations,
            "generator": generator,
            # **inference_kwargs 
        }

        # Inference
        with torch.inference_mode():
            images = base(**call_kwargs).images

        # 6. Non-blocking Save
        executor.submit(save_images_async, images, args.images_dir, example_id)
        generated_count += 1
        del images
        del payload

    # Ensure all saves finish
    executor.shutdown(wait=True)

    print(f"Generation complete. Images saved to {args.images_dir}")
    print(f"Skipped: {skipped_count}, Generated: {generated_count}")

if __name__ == "__main__":
    main()

