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

from .modules.config import DEFAULT_SEED, DEFAULT_CAUSAL_FILE, DEFAULT_INPUT_FILE


def main():
    """Main entry point for the data generation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip_causal", action="store_true")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--causal_dir", type=str, default=DEFAULT_CAUSAL_FILE)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--skip_desc", action="store_true")
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Import model functions only when needed (to handle vLLM dependencies)
    from .modules.model import initialize_model
    from .modules.pipeline import main_pipeline

    # Initialize model
    model, sampling_params = initialize_model(args.seed)

    # Execute the main pipeline
    main_pipeline(args, model, sampling_params)


if __name__ == "__main__":
    main()