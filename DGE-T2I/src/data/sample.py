import random
import os
import json
import argparse 
try:
    from .modules.config import DEFAULT_SEED, DEFAULT_OUTPUT_FILE, DATA_RAW_DIR
except ImportError:
    from modules.config import DEFAULT_SEED, DEFAULT_OUTPUT_FILE, DATA_RAW_DIR

DEFAULT_N = 20
DEFAULT_FILENAME = "qwen8b_t2i_prompts_aug_sample.json"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-path", type=str, default=DEFAULT_OUTPUT_FILE)
    argparser.add_argument("--out-path", type=str, default=os.path.join(DATA_RAW_DIR, DEFAULT_FILENAME))
    argparser.add_argument("-n", type=int, default=DEFAULT_N)
    argparser.add_argument("-s", type=int, default=DEFAULT_SEED)
    args = argparser.parse_args()

    with open(args.data_path, "r") as f: 
        data = json.load(f)
    
    sample = random.sample(range(len(data)), args.n)
    entries = []
    for idx in sample: 
        entries.append({"index": idx, **data[idx]})

    with open(args.out_path, "w") as f: 
        json.dump(entries, f, indent=2)

if __name__ == "__main__":
    main()
