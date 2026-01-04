import torch
import json
import os
import argparse
from diffusers import FluxPipeline,  FluxTransformer2DModel, GGUFQuantizationConfig
from tqdm import tqdm

MODEL_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models/FLUX1-dev"
GGUF_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models/FLUX.1-dev-gguf/flux1-dev-Q2_K.gguf"
IMAGES_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/flux"
DATA_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
EMBEDDINGS_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings/flux"
SEED = 44


def check_images_exist(example_id, output_dir, num_generations):
    """
    Check if all expected images already exist for this example.
    
    Args:
        example_id: Identifier for the example
        output_dir: Directory where images should be saved
        num_generations: Expected number of images
        
    Returns:
        True if all images exist, False otherwise
    """
    for i in range(1, num_generations + 1):
        image_path = os.path.join(output_dir, f"{example_id}-{i}.png")
        if not os.path.exists(image_path):
            return False
    return True


def generate_img_from_embeddings(
    pipeline,
    example_id,
    embedding_path,
    prompt_dir,
    num_generations=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generates images using the FLUX.1-dev pipeline with pre-encoded embeddings.
    
    Args:
        pipeline: FluxPipeline instance (loaded without text_encoder_2)
        example_id: Identifier for the example
        embedding_path: Path to the pre-encoded embeddings file
        prompt_dir: Directory to save generated images
        num_generations: Number of images to generate per prompt
        device: Device to run generation on
        
    Returns:
        The example_id if successful, None otherwise
    """
    try:
        # Load pre-encoded embeddings
        embeds = torch.load(embedding_path, map_location=device)
        prompt_embeds = embeds["prompt_embeds"].to(device)
        pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(device)
        
        # The FLUX pipeline can generate multiple images for a single prompt in one call.
        # We use a generator for reproducibility, seeded with the global SEED.
        generator = torch.Generator(device=device).manual_seed(SEED)

        # Generate images using pre-encoded embeddings
        images = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_generations,
            generator=generator
        ).images

        os.makedirs(prompt_dir, exist_ok=True)

        # Save the generated images
        for i, img in enumerate(images):
            img.save(os.path.join(prompt_dir, f"{example_id}-{i+1}.png"))

        return example_id

    except Exception as e:
        print(f"Error generating images for example {example_id} with FLUX: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using pre-encoded prompt embeddings")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to FLUX model")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=EMBEDDINGS_DIR, help="Directory containing embeddings")
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR, help="Directory to save generated images")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of images per prompt")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip examples that already have all images generated (resume mode)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load FLUX pipeline WITHOUT text encoders (since we use pre-encoded embeddings)
    print("Loading FLUX pipeline (without text encoders)...")
    transformer = FluxTransformer2DModel.from_single_file(
        GGUF_PATH,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    flux_pipeline = FluxPipeline.from_pretrained(
        args.model_path,
        text_encoder=None,  # Skip CLIP text encoder - we have pre-encoded pooled embeddings
        text_encoder_2=None,  # Skip T5 encoder - we have pre-encoded prompt embeddings
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to(device)
    # flux_pipeline.enable_model_cpu_offload()

    # Load prompts data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    # Determine range to process
    if args.start_idx is not None and args.end_idx is not None:
        # Use explicit start/end indices if provided
        start_idx = args.start_idx
        end_idx = args.end_idx
        print(f"Using explicit indices: {start_idx} to {end_idx-1}")
    else:
        # Use split parameters to compute indices
        if args.split_id >= args.num_splits:
            raise ValueError(f"split_id ({args.split_id}) must be less than num_splits ({args.num_splits})")
        if args.split_id < 0:
            raise ValueError(f"split_id ({args.split_id}) must be non-negative")
        
        total_items = len(data)
        items_per_split = total_items // args.num_splits
        remainder = total_items % args.num_splits
        
        # Distribute remainder items to first 'remainder' splits
        if args.split_id < remainder:
            start_idx = args.split_id * (items_per_split + 1)
            end_idx = start_idx + items_per_split + 1
        else:
            start_idx = args.split_id * items_per_split + remainder
            end_idx = start_idx + items_per_split
        
        print(f"Split {args.split_id}/{args.num_splits}: Processing examples {start_idx} to {end_idx-1}")
    
    data_to_process = data[start_idx:end_idx]
    
    print(f"Generating images for {len(data_to_process)} examples...")
    if args.skip_existing:
        print("Resume mode: Skipping examples with existing images")
    
    skipped_count = 0
    generated_count = 0
    
    for idx_in_slice, item in enumerate(tqdm(data_to_process)):
        idx = start_idx + idx_in_slice
        example_id = f"{idx:04d}"
        
        # Paths to pre-encoded embeddings
        prompt_embedding_path = os.path.join(
            args.embeddings_dir, f"{example_id}.pt"
        )
        
        # Generate images from prompt1 embeddings
        if os.path.exists(args.images_dir):
            # Check if prompt1 images already exist
            if args.skip_existing and check_images_exist(example_id, args.images_dir, args.num_generations):
                skipped_count += 1 # Skip example
            else:
                generate_img_from_embeddings(
                    flux_pipeline,
                    example_id,
                    prompt_embedding_path,
                    args.data_path,
                    args.num_generations,
                    device
                )
                generated_count += 1
        else:
            print(f"Warning: Embedding file not found: {prompt_embedding_path}")
    
    print(f"\nImage generation complete! Images saved to {args.images_dir}")
    if args.skip_existing:
        print(f"Skipped: {skipped_count} examples (already generated)")
        print(f"Generated: {generated_count} prompts")

