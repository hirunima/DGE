"""Configuration constants for scene graph evaluation."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

DEFAULT_PROMPTS_FILE = os.path.join(DATA_RAW_DIR, "qwen8b_t2i_prompts_aug_v1.json")
DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_RAW_DIR, "eval_v1")

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_CONCURRENT_REQUESTS = 256
MAX_MODEL_LEN = 8192
MAX_TOKENS = 1024

LABEL_THRESHOLD = 0.5