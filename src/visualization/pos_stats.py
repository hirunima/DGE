import spacy
import pandas as pd
import json
from collections import defaultdict
import numpy as np

# 1. Setup spaCy
# Keep the parser enabled for adjective-to-noun dependency links.
nlp = spacy.load("en_core_web_sm", disable=["ner"])

def analyze_benchmark(name, prompts):
    """Parses prompts and returns counts for Nouns, Adjectives, and Verbs."""
    unique_nouns = set()
    unique_adjs = set()
    unique_verbs = set()
    total_words = 0
    total_nouns = 0
    total_adjs = 0
    total_verbs = 0
    total_adj_per_noun = 0.0

    print(f"Processing {name}...")
    for doc in nlp.pipe(prompts, batch_size=256):
        adj_mod_counts = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            
            total_words += 1
            pos = token.pos_
            lemma = token.lemma_.lower()
            
            if pos == "NOUN" or pos == "PROPN": # Including Proper Nouns for benchmarks
                unique_nouns.add(lemma)
                total_nouns += 1
                adj_mod_count = 0
                for child in token.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        adj_mod_count += 1
                adj_mod_counts.append(adj_mod_count)
            elif pos == "ADJ":
                unique_adjs.add(lemma)
                total_adjs += 1
            elif pos == "VERB":
                unique_verbs.add(lemma)
                total_verbs += 1

        total_adj_per_noun += np.mean(adj_mod_counts)

    total_prompts = len(prompts)
    avg_nouns = total_nouns / total_prompts if total_prompts else 0
    avg_adjs = total_adjs / total_prompts if total_prompts else 0
    avg_verbs = total_verbs / total_prompts if total_prompts else 0
    avg_adj_per_noun = total_adj_per_noun / total_prompts if total_prompts else 0

    return {
        "Benchmark": name,
        "Avg Adjs/Prompt": avg_adjs,
        "Avg Adjs/Noun/Prompt": avg_adj_per_noun
    }

# 2. Load your data (Replace paths with your actual local paths)
# --- EvalMuse ---
# Assuming evalmuse_df is your loaded dataframe
# evalmuse_prompts = evalmuse_df['caption'].tolist()

# --- TIIF-Bench ---
tiif_bench_prompts = [json.loads(line)['short_description'] for line in open('/fs/nexus-projects/scene_graph_sd/TIIF-Bench/data/test_prompts/all_prompts.jsonl')]

tiif_bench_prompts_l = [json.loads(line)['long_description'] for line in open('/fs/nexus-projects/scene_graph_sd/TIIF-Bench/data/test_prompts/all_prompts.jsonl')]


# --- DPGBench ---
dpg_prompts = [json.loads(line)['prompt'] for line in open('/fs/nexus-projects/scene_graph_sd/ELLA/dpg_bench/prompts.jsonl')]

# --- GenEval ---
# geneval_prompts = [json.loads(line)['prompt'] for line in open('geneval.jsonl')]

our_prompts = [obj['prompt'] for obj in json.load(open('/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json'))]

# 3. Execution & Comparison
# (Dummy list for demonstration; replace with actual loaded lists)
benchmarks = {
    "TIIF-Bench": tiif_bench_prompts, 
    "TIIF-Bench (long)": tiif_bench_prompts_l, 
    "DPGBench": dpg_prompts, 
    # "GenEval": geneval_prompts, 
    "Ours": our_prompts, 
}

results = []
for name, prompts in benchmarks.items():
    results.append(analyze_benchmark(name, prompts))

# 4. Display Results
df_results = pd.DataFrame(results)
print("\n--- POS Analysis Comparison ---")
print(df_results.to_string(index=False))
