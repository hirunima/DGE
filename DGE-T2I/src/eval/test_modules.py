import argparse

try:
    from .modules.config import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROMPTS_FILE,
        MAX_TOKENS,
    )
except ImportError:
    from modules.config import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROMPTS_FILE,
        MAX_TOKENS,
    )

print(DEFAULT_OUTPUT_DIR)
