"""Module for handling LLM model operations and prompt generation."""

import json
from typing import List, Dict, Any
from dataclasses import asdict
from vllm import LLM, EngineArgs, SamplingParams
from torch.cuda import is_available as cuda_available, device_count as gpu_count
from .config import (
    MODEL_NAME, MAX_CONCURRENT_REQUESTS, MAX_MODEL_LEN, MAX_TOKENS,
    TEMPERATURE, TOP_K, TOP_P, FILTER_MAX_TOKENS
)


def initialize_model(seed: int = 0):
    """Initialize the LLM model with appropriate configuration."""
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
        disable_log_stats=True
    )

    engine_args = asdict(engine_args) | {
        "seed": seed,
        "mm_processor_cache_gb": 4,
        "tensor_parallel_size": gpu_count() if cuda_available() else 1,
    }
    model = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop_token_ids=None
    )
    return model, sampling_params


def generate_reasoning_removed(model, prompts, reasoning_file=None, **kwargs):
  results = model.generate(prompts, **kwargs)
  file_output = {}
  for res in results:
    res.outputs[0].text = res.outputs[0].text.split("</think>")[-1]
    if reasoning_file:
      file_output[res.request_id] = {
          "prompt": res.prompt,
          "reasoning_removed_output": res.outputs[0].text
      }
  
  if reasoning_file:
    with open(reasoning_file, "w") as f:
      json.dump(file_output, f, indent=6)
    del file_output
  return results



def get_description_prompts(args, prompts, json_metadata):
    """Process prompts to generate descriptions based on causal contexts."""
    desc_prompt = """You are evaluating an text-to-image generation model. The model was given the following prompt: "{obs}".
Describe, in one, short sentence, some key visual details the generated image should have that weren't mentioned in the prompt.
*DO NOT use any introductory phrases** like "The generated image should show" or "A detail is" in your description. The output
must be ONLY the descriptive sentence.

Here are some examples:

Prompt: "A chef enjoying her vacation."
Description: "A woman is in her causal clothes, The woman is not in a kitchen."

Prompt: "A peacock sleeping."
Description: "A peacock has its feathers closed."

Prompt: "An apple tree in spring."
Description: "A tree has white flowers in full bloom."

Prompt: {obs}
Description:
    """
    desc_prompts, prompt_metadata = [], []
    omitted_cnt = 0
    for i, prompt_result in enumerate(prompts):
        try:
            if not args.skip_causal:
                output_text = prompt_result.outputs[0].text
                if output_text.startswith("```json"):
                    output_text = output_text[7:]
                elif output_text.startswith("```"):
                    output_text = output_text[3:]
                if output_text.endswith("```"):
                    output_text = output_text[:-3]
                prompt_pairs = json.loads(output_text)
            else:
                prompt_pairs = json.loads(prompt_result)
            for dtype in ["positive", "negative"]:
                question = desc_prompt.format(obs=prompt_pairs[dtype]["context"])
                prompt = (
                    "SYSTEM\nYou are a helpful assistant.\n"
                    f"USER\n{question}\n"
                    f"ASSISTANT\n"
                )
                inputs = {
                    "prompt": prompt
                }
                desc_prompts.append(inputs)
            prompt_metadata.append({
                "prompt1": prompt_pairs["positive"]["context"],
                "prompt2": prompt_pairs["negative"]["context"],
                "filename": json_metadata[i]["filename"]
            })
        except Exception as e:
            print(e)
            omitted_cnt += 1
    print("Omitted ", omitted_cnt, "prompt pairs when generating descriptions.")
    return desc_prompts, prompt_metadata


def filter_results(results):
    """Prepare filter prompts to validate result quality."""
    filter_prompts = []
    q1 = "Does the description just restate the prompt? Answer Yes. or No."
    q2 = "Could an image not matching the description comply with the prompt? Answer Yes. or No."
    for res in results:
        try:
            pairs = [
                (res.get("prompt1", ""), res.get("description1", {})),
                (res.get("prompt2", ""), res.get("description2", {}))
            ]
            for prompt_text, desc_text in pairs:
                if prompt_text and desc_text:
                    prompt1 = (
                        "SYSTEM\nYou are a helpful assistant.\n"
                        f'USER\nPrompt: "{prompt_text}"\nDescription: "{desc_text}"\n\nQuestion: {q1}\n'
                        f"ASSISTANT\n"
                    )
                    prompt2 = (
                        "SYSTEM\nYou are a helpful assistant.\n"
                        f'USER\nPrompt: "{prompt_text}"\nDescription: "{desc_text}"\n\nQuestion: {q2}\n'
                        f"ASSISTANT\n"
                    )
                    filter_prompts.append({"prompt": prompt1})
                    filter_prompts.append({"prompt": prompt2})
        except Exception as e:
            print(f"Error preparing filter prompts: {e}")
    return filter_prompts