import argparse

def main():
    parser = argparse.ArgumentParser(description="Encode prompts to embeddings for T2I generation")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to prompts JSON file")
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Directory to save embeddings")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON key containing the prompt text")
    parser.add_argument("--device-map", type=str, default="balanced", help="Device map for model parallelism (e.g., balanced, auto, sequential, none)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing (overrides split params)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (overrides split params)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for parallel processing")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for this process (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts with existing embeddings")
    args = parser.parse_args()

    print(f"Encoding not implemented.")


if __name__ == "__main__":
    main()
