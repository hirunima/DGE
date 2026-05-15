import json
import argparse
from collections import defaultdict
import random
import os
import sys
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ..modules.config import DEFAULT_OUTPUT_DIR, SAMPLE_DIFF, N_PROMPTS, DEFAULT_IMAGES_DIR, DEFAULT_PROMPTS_FILE
except ImportError:
    from modules.config import DEFAULT_OUTPUT_DIR, SAMPLE_DIFF, N_PROMPTS, DEFAULT_IMAGES_DIR, DEFAULT_PROMPTS_FILE

def main(): 
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    # gather scores of all models

    all_scores = {}
    indices = None
    
    print("Found files: ")
    for filename in os.listdir(DEFAULT_OUTPUT_DIR): 
        if filename.endswith("_eval_summary.json"):
            print(filename)
            
            with open(os.path.join(DEFAULT_OUTPUT_DIR, filename), "r") as f: 
                summary = json.load(f)

            if indices == None: indices = set(summary["image_scores"].keys())
            else: indices = indices & set(summary["image_scores"].keys())

            model_name = filename.rsplit("_eval_summary.json")[0]
            all_scores[model_name] = summary["image_scores"]
    # prompts to exclude
    indices = [idx for idx in indices if int(idx) not in [62, 2622, 99]]
            
    # get all prompts for reference
    with open(DEFAULT_PROMPTS_FILE) as f: 
        all_prompts = json.load(f)

    # get random set of prompts
    selected_pids = random.sample(indices, min(len(indices), N_PROMPTS))

    
    selected_images = []
    for idx in selected_pids:
        # randomly pick two models
        all_models = list(all_scores.keys())
        model1, model2 = random.sample(all_models, 2)

        # get all pairs of scores satisfying score diff
        filtered_images = []
        for i, score1 in enumerate(all_scores[model1][idx]["scores"]):
            for j, score2 in enumerate(all_scores[model2][idx]["scores"]): 
                if abs(score2["overall"] - score1["overall"]) < SAMPLE_DIFF: 
                    filtered_images.append({
                        "prompt_index": idx,
                        "prompt": all_prompts[int(idx)]["prompt"],
                        "model1": model1, 
                        "model2": model2, 
                        "model1_idx": i, 
                        "model2_idx": j
                    })
        # choose 1 (maybe make this a param?)
        if len(filtered_images) > 0: selected_images.append(random.choice(filtered_images))

    os.makedirs(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "image1"),exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "image2"),exist_ok=True)

    selected_prompts = {}
    models1, models2 = {}, {}
    for metadata in selected_images: 
        idx = int(metadata["prompt_index"])
        i = metadata["model1_idx"] 
        j = metadata["model2_idx"] 
        model1, model2 = random.sample([metadata["model1"], metadata["model2"]], 2) # maybe not necessary?
        image1 = Image.open(os.path.join(DEFAULT_IMAGES_DIR, model1, f"{idx:04d}-{i+1}.png"))
        image2 = Image.open(os.path.join(DEFAULT_IMAGES_DIR, model2, f"{idx:04d}-{j+1}.png"))
        image1.save(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "image1", f"{idx:04d}-{i+1}.png"))
        image2.save(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "image2", f"{idx:04d}-{j+1}.png"))
        models1[f"{idx:04d}-{i+1}.png"] = model1
        models2[f"{idx:04d}-{j+1}.png"] = model2
        selected_prompts[f"{idx:04d}"] = metadata["prompt"]

    with open(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "prompts.json"), "w") as f: 
        json.dump(selected_prompts, f, indent=2)

    with open(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "image1", "models.json"), "w") as f: 
        json.dump(models1, f, indent=2)

    with open(os.path.join(DEFAULT_IMAGES_DIR, "survey_samples", "image2", "models.json"), "w") as f: 
        json.dump(models2, f, indent=2)
        
      
if __name__ == "__main__":
    main()

