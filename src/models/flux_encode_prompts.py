import torch
import json
import os
import argparse
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

MODEL_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models/FLUX1-dev"
DATA_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
EMBEDDINGS_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings/flux"

def encode_prompts(prompts, text_encoder, tokenizer, text_encoder_2, tokenizer_2, device, max_length=512):
    """
    Encode a list of prompts using both CLIP and T5 encoders.
    
    Args:
        prompts: List of text prompts to encode
        text_encoder: CLIPTextModel instance (for pooled embeddings)
        tokenizer: CLIPTokenizer instance
        text_encoder_2: T5EncoderModel instance (for prompt embeddings)
        tokenizer_2: T5TokenizerFast instance
        device: Device to run encoding on
        max_length: Maximum sequence length for T5
        
    Returns:
        Dictionary containing prompt_embeds (from T5) and pooled_prompt_embeds (from CLIP)
    """
    # Encode with CLIP (text_encoder) for pooled embeddings
    clip_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,  # CLIP max length
        truncation=True,
        return_tensors="pt",
    )
    clip_input_ids = clip_inputs.input_ids.to(device)
    
    with torch.no_grad():
        clip_outputs = text_encoder(clip_input_ids, output_hidden_states=True)
        # Use pooled output from CLIP
        pooled_prompt_embeds = clip_outputs.pooler_output
    
    # Encode with T5 (text_encoder_2) for prompt embeddings
    t5_inputs = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    t5_input_ids = t5_inputs.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds = text_encoder_2(t5_input_ids)[0]
    
    return {
        "prompt_embeds": prompt_embeds.cpu(),  # From T5
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),  # From CLIP
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode prompts using T5 text encoder")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to FLUX model")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to prompts JSON file")
    parser.add_argument("--output_dir", type=str, default=EMBEDDINGS_DIR, help="Directory to save embeddings")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP text encoder and tokenizer (for pooled embeddings)
    print("Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer"
    )
    
    # Load T5 text encoder and tokenizer (for prompt embeddings)
    print("Loading T5 text encoder...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.model_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        args.model_path,
        subfolder="tokenizer_2"
    )
    
    # Load prompts data
    print(f"Loading prompts from {args.data_path}...")
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    # Validate split parameters
    if args.split_id >= args.num_splits:
        raise ValueError(f"split_id ({args.split_id}) must be less than num_splits ({args.num_splits})")
    if args.split_id < 0:
        raise ValueError(f"split_id ({args.split_id}) must be non-negative")
    
    # Compute start and end indices for this split
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
    
    data_to_process = data[start_idx:end_idx]
    
    print(f"Split {args.split_id}/{args.num_splits}: Processing examples {start_idx} to {end_idx-1} ({len(data_to_process)} items)")
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "prompt1_embeddings"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt2_embeddings"), exist_ok=True)
    
    # Encode prompts for this split
    print(f"Encoding {len(data_to_process)} prompt pairs with CLIP and T5...")
    for idx_in_slice, item in enumerate(tqdm(data_to_process)):
        idx = start_idx + idx_in_slice
        prompt = item["prompt"]
        example_id = f"{idx:04d}"
        
        # Encode prompt1 with both CLIP and T5
        embeds = encode_prompts(
            [prompt], text_encoder, tokenizer, text_encoder_2, tokenizer_2, device, args.max_length
        )
        torch.save(
            embeds,
            os.path.join(args.output_dir, f"{example_id}.pt")
        )
    
    print(f"All embeddings saved to {args.output_dir}")
    print("You can now use flux_generate_from_embeddings.py to generate images from these embeddings.")

