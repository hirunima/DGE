# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import argparse
import gc  # <--- FIX 1: Import Garbage Collection
from safetensors.torch import load_file
from tqdm import tqdm

import torch

# Allow running this file directly by adding the BAGEL package root to sys.path.
BAGEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "BAGEL"))
if BAGEL_ROOT not in sys.path:
    sys.path.insert(0, BAGEL_ROOT)
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def resolve_compute_dtype(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input

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


def get_kv_and_latent(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure gen_model is accessible
    if 'gen_model' not in globals():
        raise RuntimeError("gen_model is not defined in global scope")

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(resolution, resolution)] * num_images, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    return past_key_values, generation_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated images (legacy mode).")
    parser.add_argument("--data_path", type=str, default=None, help="JSON file containing prompt items.")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory to save generated images.")
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Unused (compatibility with job scripts).")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON key containing the prompt text.")
    parser.add_argument("--num_generations", type=int, default=None, help="Number of images per prompt.")
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--model_id", type=str, default=None, help="Model local path.")
    parser.add_argument("--device-map", type=str, default="none", help="Unused (compatibility with job scripts).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params).")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params).")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing.")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed).")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts that already have images.")
    args = parser.parse_args()
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # --- FIX 2: Define Device BEFORE loading model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_id, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_id, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_id, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    
    model = Bagel(language_model, None, config)
    
    # Move empty model to device first
    model = model.to(device)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_id)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    print("Loading model weights...")
    model_state_dict_path = os.path.join(args.model_id, "ema.safetensors")
    
    # Load weights directly to target device to save CPU RAM
    model_state_dict = load_file(model_state_dict_path, device=device)
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    
    # Cleanup weights immediately
    del model_state_dict
    gc.collect() 
    if device == "cuda":
        torch.cuda.empty_cache()

    model = model.eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    with open(args.data_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    
    os.makedirs(args.embeddings_dir, exist_ok=True)

    start, end = resolve_indices(len(data), args.start_idx, args.end_idx, args.num_splits, args.split_id)
    
    print(f"Processing items {start} to {end}...")

    # Using tqdm for progress bar
    for idx in tqdm(range(start, end)):
        try:
            entry = data[idx]
            prompt = entry['prompt']
            example_id = f"{idx:04d}"

            if args.skip_existing and os.path.exists(os.path.join(args.embeddings_dir,  f"{example_id}.pt" )):
                continue

            kv, latent_info = get_kv_and_latent(
                prompt=prompt,
                cfg_scale=cfg_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                num_images=1,
                resolution=args.resolution,
                device=device,
            )
            
            # Save the file
            save_path = os.path.join(args.embeddings_dir, f"{example_id}.pt")
            torch.save({"kv": kv, "latent_info": latent_info}, save_path)
            
            # --- FIX 3: CRITICAL LOOP CLEANUP ---
            # Python holds these variables in memory unless explicitly deleted, 
            # causing RAM usage to grow with every loop iteration.
            del kv
            del latent_info
            
            # Force Python to actually free the memory NOW, not later.
            gc.collect() 
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            # Ensure cleanup happens even on error
            if 'kv' in locals(): del kv
            if 'latent_info' in locals(): del latent_info
            gc.collect()
            continue

    print(f"Completed tasks.")