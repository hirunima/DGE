"""
Modular data generation pipeline for scene graph processing.

This module serves as the entry point for the data generation pipeline.
All functionality is delegated to specialized modules in the modules/ directory.
"""

import argparse
import sys
import os
import random
import numpy as np
import torch

from modules.config import DEFAULT_SEED, DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_FILE


def main():
    """Main entry point for the data generation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    # Import model functions only when needed (to handle vLLM dependencies)
    from modules.model import initialize_model
    from modules.pipeline import main_pipeline

    # Initialize model
    model, sampling_params = initialize_model(args.seed)

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Execute the main pipeline
    main_pipeline(args, model, sampling_params)


if __name__ == "__main__":
    main()