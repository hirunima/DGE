import json
import argparse
from collections import defaultdict
import random
import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.config import DEFAULT_OUTPUT_DIR, SAMPLE_DIFF, SAMPLE_N, DEFAULT_IMAGES_DIR, DEFAULT_PROMPTS_FILE

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(DEFAULT_OUTPUT_DIR, f"{args.model1}_eval_summary.json"), "r") as f: 
        summary1 = json.load(f)

    with open(os.path.join(DEFAULT_OUTPUT_DIR, f"{args.model2}_eval_summary.json"), "r") as f: 
        summary2 = json.load(f)
    
    with open(DEFAULT_PROMPTS_FILE) as f: 
        all_prompts = json.load(f)
    
    indices = summary1["image_scores"].keys() & summary2["image_scores"].keys() 
    indices = [idx for idx in indices if int(idx) not in [62, 2622, 99]]
    filtered_images = []
    for idx in indices:
        scores1 = sorted(enumerate([score["overall"] for score in summary1["image_scores"][idx]["scores"]]), key=lambda x: x[1])
        scores2 = sorted(enumerate([score["overall"] for score in summary2["image_scores"][idx]["scores"]]), key=lambda x: x[1])
        j = 0
        for i, score in scores1:
            while j < len(scores2) and abs(scores2[j][1] - score) < SAMPLE_DIFF: 
                j += 1
            if j < len(scores2): 
                filtered_images.append({
                    "prompt_index": idx,
                    "prompt": all_prompts[int(idx)]["prompt"],
                    args.model1: i, 
                    args.model2: scores2[j][0]
                })
                j += 1
    
    selected_images = random.sample(filtered_images, SAMPLE_N)

    os.makedirs(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "model1"),exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "model2"),exist_ok=True)

    prompts = {}
    for metadata in selected_images: 
        idx = int(metadata["prompt_index"])
        i = metadata[args.model1] 
        j = metadata[args.model2] 
        image1 = Image.open(os.path.join(DEFAULT_IMAGES_DIR, args.model1, f"{idx:04d}-{i+1}.png"))
        image2 = Image.open(os.path.join(DEFAULT_IMAGES_DIR, args.model2, f"{idx:04d}-{j+1}.png"))
        image1.save(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "model1", f"{idx:04d}-{i+1}.png"))
        image2.save(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "model2", f"{idx:04d}-{j+1}.png"))
        prompts[f"{idx:04d}-{i+1}.png"] = metadata["prompt"]

    with open(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "prompts.json"), "w") as f: 
        json.dump(prompts, f, indent=2)
        
      
if __name__ == "__main__":
    main()


