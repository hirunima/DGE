from vllm_omni.entrypoints.omni import Omni

DEFAULT_DATA_PATH = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_EMBEDDINGS_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings"
DEFAULT_IMAGES_DIR = "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images"
DEFAULT_SEED = 44

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
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism (e.g., balanced, auto, sequential, none)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts that already have images")
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx, end_idx = resolve_indices(len(data), args.start_idx, args.end_idx, args.num_splits, args.split_id)
    
    prompts = [item.get("prompt") for item in data[start_idx:end_idx] for _ in range(args.num_generations)]
    omni = Omni(model="Qwen/Qwen-Image-2512")

    outputs = omni.generate(prompts)
    images = outputs[0].request_output[0].images
    for idx in tqdm(range(start_idx, end_idx)): 
        for i in range(arg.num_generations): 
            image[(idx - start_idx) * arg.num_generations + i].save(os.path.join(args.images_dir, f"{idx:04d}-{i + 1}.png"))

if __name__ == "__main__":
    main()