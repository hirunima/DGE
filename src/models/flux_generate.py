import torch
import json
import os
import argparse
from diffusers import FluxPipeline
from tqdm import tqdm

MODEL_PATH="checkpoints/FLUX1-dev"
IMAGES_DIR="flux_ovad_t2i"
DATA_PATH="T2I-R1/src/t2i-r1/src/infer/commonsense_ovad/description_qwen4bg_prompt_pairs_sam_unfiltered.json"
SEED=44

def generate_img(pipeline, example_id, prompt, prompt_dir, num_generations=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generates images using the FLUX.1-dev pipeline.
    The `cot_prompt` and `processor` arguments are kept for compatibility with the calling
    function's signature but are not used by this implementation.
    The `pipeline` argument is expected to be an instance of diffusers.FluxPipeline.
    """
    try:
        # The FLUX pipeline can generate multiple images for a single prompt in one call.
        # We use a generator for reproducibility, seeded with the global SEED.
        generator = torch.Generator(device=device).manual_seed(SEED)

        images = pipeline(
            prompt,
            num_images_per_prompt=num_generations,
            generator=generator
        ).images

        os.makedirs(prompt_dir, exist_ok=True)

        # Save the generated images
        for i, img in enumerate(images):
            img.save(os.path.join(prompt_dir, f"{example_id}-{i+1}.png"))

        # The FLUX pipeline does not modify the input prompt.
        # We return the original prompt for compatibility with the calling function's logic.
        return prompt

    except Exception as e:
        print(f"Error generating images for example {example_id} with FLUX: {e}")
        return None



if __name__ == "__main__":
    flux_pipeline = FluxPipeline.from_pretrained(
        MODEL_PATH,
        text_encoder_2=None,
        torch_dtype=torch.bfloat16, 
        device_map="balanced"
    )
    device = "cuda"

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    for idx, item in enumerate(tqdm(data)):
        prompt1 = item["prompt1"]
        prompt2 = item["prompt2"]
        example_id = f"{idx:04d}"
        prompt1_dir = os.path.join(IMAGES_DIR, "prompt1_img", "original")
        prompt2_dir = os.path.join(IMAGES_DIR, "prompt2_img", "original")
        generate_img(flux_pipeline, example_id, prompt1, prompt1_dir, 5, device)
        generate_img(flux_pipeline, example_id, prompt2, prompt2_dir, 5, device)