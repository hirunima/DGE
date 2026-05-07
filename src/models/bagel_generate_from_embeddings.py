# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import argparse
from safetensors.torch import load_file
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

import torch
import torch.distributed as dist

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def should_setup_distributed():
    if not torch.cuda.is_available():
        return False
    if os.environ.get("WORLD_SIZE"):
        return int(os.environ["WORLD_SIZE"]) > 1
    return "LOCAL_RANK" in os.environ


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


def load_prompts(data_path, metadata_file, prompt_key):
    if data_path:
        with open(data_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError("Expected data_path to contain a list of prompt items.")
        return data

    if metadata_file:
        with open(metadata_file, "r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp]

    raise ValueError("Either data_path or metadata_file must be provided.")


def check_images_exist(example_id, output_dir, num_generations):
    for i in range(1, num_generations + 1):
        image_path = os.path.join(output_dir, f"{example_id}-{i}.png")
        if not os.path.exists(image_path):
            return False
    return True


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None):  # 添加device参数
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

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated images (legacy mode).")
    parser.add_argument("--metadata_file", type=str, default=None, help="JSONL file containing metadata per prompt (legacy mode).")
    parser.add_argument("--data_path", type=str, default=None, help="JSON file containing prompt items.")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory to save generated images.")
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Unused (compatibility with job scripts).")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON key containing the prompt text.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=None, help="Number of images per prompt.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--model_id", type=str, default=None, help="Hugging Face model ID or local path.")
    parser.add_argument("--model-path", type=str, default="hf/BAGEL-7B-MoT/")
    parser.add_argument("--device-map", type=str, default="none", help="Unused (compatibility with job scripts).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params).")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params).")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing.")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed).")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts that already have images.")
    args = parser.parse_args()

    if args.images_dir and not args.data_path:
        parser.error("--images_dir requires --data_path.")
    if not args.images_dir and not args.output_dir:
        parser.error("Provide --images_dir (recommended) or --output_dir (legacy mode).")
    if not args.data_path and not args.metadata_file:
        parser.error("Provide --data_path or --metadata_file.")

    if args.num_generations is None:
        args.num_generations = args.num_images
    else:
        args.num_images = args.num_generations
    if args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer.")

    model_path = args.model_path
    if args.model_id:
        model_path = args.model_id
    
    seed = args.seed
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Should setup distributed: ", should_setup_distributed)
    if should_setup_distributed():
        setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = args.images_dir or args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(model_path, "ema.safetensors")
    use_offload = torch.cuda.is_available() and world_size == 1
    if use_offload:
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "20GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_state_dict_path,
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    else:
        model_state_dict = load_file(model_state_dict_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        if rank == 0:
            print(msg)
        del model_state_dict
        model = model.to(device).eval()

    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    metadatas = load_prompts(args.data_path, args.metadata_file, args.prompt_key)
    total_metadatas = len(metadatas)

    if args.start_idx is not None or args.end_idx is not None or args.num_splits > 1 or args.split_id != 0:
        start, end = resolve_indices(total_metadatas, args.start_idx, args.end_idx, args.num_splits, args.split_id)
    else:
        prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
        start = rank * prompts_per_gpu
        end = min(start + prompts_per_gpu, total_metadatas)

    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        metadata = metadatas[idx]
        prompt = metadata.get(args.prompt_key)
        if not prompt:
            continue

        if args.images_dir:
            example_id = f"{idx:04d}"
            if args.skip_existing and check_images_exist(example_id, output_dir, args.num_generations):
                print(f"GPU {rank} skipping prompt {idx} (already generated)")
                continue
        else:
            outpath = os.path.join(output_dir, f"{idx:0>5}")
            os.makedirs(outpath, exist_ok=True)

        if seed is not None:
            seed_value = seed + idx
            import random
            import numpy as np
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value)
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        if not args.images_dir:
            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)

            flag = True
            for sample_idx in range(args.num_images):
                if not os.path.exists(os.path.join(sample_path, f"{sample_idx:05}.png")):
                    flag = False
                    break
            if flag:
                print(f"GPU {rank} skipping generation for prompt: {prompt}")
                continue

            with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
                json.dump(metadata, fp)

        image_list = []

        remaining = args.num_generations
        while remaining > 0:
            current_batch = min(args.batch_size, remaining)
            tmp_image_list = generate_image(
                prompt=prompt,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                num_images=current_batch,
                resolution=args.resolution,
                device=device,
            )
            image_list.extend(tmp_image_list)
            remaining -= current_batch

        if args.images_dir:
            for i, sample in enumerate(image_list):
                sample = sample.crop(sample.getbbox())
                sample.save(os.path.join(output_dir, f"{example_id}-{i + 1}.png"))
        else:
            sample_count = 0
            for sample in image_list:
                sample = sample.crop(sample.getbbox())
                sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1

    print(f"GPU {rank} has completed all tasks")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
