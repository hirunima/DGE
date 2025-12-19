"""Module for handling the main pipeline workflow."""

import json
import os
from pathlib import Path
from typing import Tuple
from .config import DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_FILE, DEFAULT_CAUSAL_FILE
from .processing import process_data
from .model import generate_reasoning_removed, get_description_prompts, filter_results
from vllm import SamplingParams


def handle_causal_processing(args, model, sampling_params, input_file: str):
    """Handle the causal processing step."""
    if not args.skip_causal:
        causal_prompts, img_metadata = process_data(input_file, sample=args.samples)
        causal_titles = generate_reasoning_removed(
            model, causal_prompts, sampling_params=sampling_params, use_tqdm=True
        )
        objs = []
        for i, desc in enumerate(causal_titles):
            try:
                obj = json.loads(desc.outputs[0].text)
                obj["filename"] = img_metadata[i]
                objs.append(obj)
            except:
                continue
        with open(args.causal_dir, 'w') as f:
            json.dump(objs, f, indent=6)
        return causal_titles, img_metadata
    else:
        causal_titles = json.load(open(args.causal_dir))
        # Note: If skipping causal, we need to load metadata separately
        # For now, assuming we have json_metadata already available
        img_metadata = [{"filename": f"file_{i}" for i in range(len(causal_titles))}]  # Placeholder
        return causal_titles, img_metadata


def handle_description_processing(args, model, sampling_params, causal_titles, img_metadata):
    """Handle the description processing and filtering step."""
    from .config import FILTER_MAX_TOKENS

    if not args.skip_desc:
        # Prepare and generate causal prompts
        desc_prompts, prompt_metadata = get_description_prompts(args, causal_titles, img_metadata)
        descriptions = generate_reasoning_removed(
            model, desc_prompts, sampling_params=sampling_params, use_tqdm=True
        )

        # Combine descriptions and causal titles
        results = []
        for i in range(0, len(desc_prompts), 2):
            try:
                results.append({
                    "prompt1": prompt_metadata[i // 2]["prompt1"],
                    "description1": descriptions[i].outputs[0].text.strip(),
                    "prompt2": prompt_metadata[i // 2]["prompt2"],
                    "description2": descriptions[i+1].outputs[0].text.strip()
                })
            except Exception as e:
                print(f"Error combining results: {e}")

        # Filter the results
        filter_prompts = filter_results(results)
        filter_sampling_params = SamplingParams(
            max_tokens=FILTER_MAX_TOKENS,
            stop_token_ids=None
        )
        filter_answers = generate_reasoning_removed(
            model, filter_prompts, sampling_params=filter_sampling_params, use_tqdm=True
        )

        final_results = []
        bad_results = []
        ans_idx = 0
        for i, res in enumerate(results):
            ans1 = filter_answers[ans_idx].outputs[0].text.strip()
            ans2 = filter_answers[ans_idx+1].outputs[0].text.strip()
            ans3 = filter_answers[ans_idx+2].outputs[0].text.strip()
            ans4 = filter_answers[ans_idx+3].outputs[0].text.strip()
            ans_idx += 4

            if not ans1.lower().startswith("yes") and not ans2.lower().startswith("yes") and not ans3.lower().startswith("yes") and not ans4.lower().startswith("yes"):
                final_results.append(res)
            else:
                res["ans1"] = ans1
                res["ans2"] = ans2
                res["ans3"] = ans3
                res["ans4"] = ans4
                bad_results.append(res)
        return final_results, bad_results
    else:
        return [], []


def main_pipeline(args, model, sampling_params):
    """Execute the main pipeline workflow."""
    # Configuration
    input_file = args.input_file
    output_file = DEFAULT_OUTPUT_FILE
    unfiltered_output_file = os.path.splitext(output_file)[0] + "_unfiltered.json"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Handle causal processing
    causal_titles, img_metadata = handle_causal_processing(args, model, sampling_params, input_file)

    # Handle description processing
    final_results, bad_results = handle_description_processing(args, model, sampling_params, causal_titles, img_metadata)

    # Save the final results
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=6)

    with open(unfiltered_output_file, 'w') as f:
        json.dump(bad_results, f, indent=6)

    print(f"Finished. Saved {len(final_results)} results to {output_file}")
    return final_results, bad_results