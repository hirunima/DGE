"""
Modular scene graph evaluation pipeline for generated images.
"""

import argparse

try:
    from .modules.config import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROMPTS_FILE,
        DEFAULT_IMAGES_DIR,
        MAX_TOKENS,
    )
except ImportError:
    from modules.config import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROMPTS_FILE,
        DEFAULT_IMAGES_DIR,
        MAX_TOKENS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated images against scene graphs with Qwen3-VL.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--sg_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--image_pattern", type=str, default="{index:04d}-{generation}.png")
    parser.add_argument("--generation", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    try:
        from .modules.model import initialize_model
        from .modules.pipeline import main_pipeline
    except ImportError:
        from modules.model import initialize_model
        from modules.pipeline import main_pipeline

    if args.prompts_file == "None": args.prompts_file = None

    model, sampling_params = initialize_model(args.seed)
    main_pipeline(args, model, sampling_params)


if __name__ == "__main__":
    main()
