"""
Module initialization for the data generation pipeline.

This file exposes the main functions needed by the pipeline.
Note: Some modules require vLLM which may not be available in all environments.
"""

from .config import *
# Only import non-vLLM dependent modules by default
from .processing import generate_question, process_data

# The following imports require vLLM and should be imported selectively when needed:
# from .model import initialize_model, generate_reasoning_removed, get_description_prompts, filter_results
# from .pipeline import main_pipeline, handle_causal_processing, handle_description_processing