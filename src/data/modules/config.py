"""Configuration constants for the data generation pipeline."""

import os

# Calculate project root - go up 4 levels from this file to get to the project root
# This file is at: project_root/src/data/modules/config.py
# So we need to go up 4 levels: config.py -> modules -> data -> src -> project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Configuration constants
DEFAULT_INPUT_FILE = "/fs/nexus-projects/scene_graph_sd/ovad/ovad_test_attribute_filtered.json"
DEFAULT_OUTPUT_FILE = os.path.join(DATA_RAW_DIR, "qwen8b_t2i_prompts.json")
DEFAULT_SEED = 42

# Model configuration
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_CONCURRENT_REQUESTS = 256
MAX_MODEL_LEN = 8192 #35000
MAX_TOKENS = 8192#25000
FILTER_MAX_TOKENS = 100

# Sampling parameters
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 1.0


# Object sampling parameters
THRESHOLD = 5
SCORE_THRESHOLD = 0.6
GAMMA = 1
BETA = 1
MAX_ITEMS_PER_SCENE = 3
# MAX_ATTRIBUTES_PER_ITEM = 2