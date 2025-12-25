"""Module for handling the main pipeline workflow."""

import json
import os
from pathlib import Path
from typing import Tuple
from .config import DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_FILE
from .processing import process_data
from .model import generate_reasoning_removed, get_description_prompts, filter_results
from vllm import SamplingParams


def generate_t2i_prompts(args, model, sampling_params, input_file: str):
    prompts, img_filenames = process_data(input_file, sample=args.samples)
    outs = generate_reasoning_removed(model, prompts, sampling_params=sampling_params, use_tqdm=True)
    t2i_prompts = [{"prompt": outs.outputs[0].text, "filename": img_metadata[i]} for out in outs]
    return t2i_prompts

def main_pipeline(args, model, sampling_params):
    """Execute the main pipeline workflow."""
    # Configuration
    input_file = args.input_file
    output_file = DEFAULT_OUTPUT_FILE
    unfiltered_output_file = os.path.splitext(output_file)[0] + "_unfiltered.json"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Handle causal processing
    results = generate_t2i_prompts(args, model, sampling_params, input_file)

    # Save the final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=6)

    print(f"Finished. Saved {len(results)} results to {output_file}")