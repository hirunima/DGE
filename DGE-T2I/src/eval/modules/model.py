"""Module for handling VLM model initialization."""

from dataclasses import asdict
from typing import Tuple

from torch.cuda import is_available as cuda_available, device_count as gpu_count
from vllm import LLM, EngineArgs, SamplingParams

from .config import (
    MODEL_NAME,
    MAX_CONCURRENT_REQUESTS,
    MAX_MODEL_LEN,
    MAX_TOKENS,
)


def initialize_model(seed: int = 0) -> Tuple[LLM, SamplingParams]:
    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_CONCURRENT_REQUESTS,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
    )
    engine_args = asdict(engine_args) | {
        "seed": seed,
        "mm_processor_cache_gb": 4,
        "tensor_parallel_size": gpu_count() if cuda_available() else 1,
    }
    model = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        stop_token_ids=None,
        seed=seed,
    )
    return model, sampling_params
