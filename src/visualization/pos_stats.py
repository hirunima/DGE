import spacy
import pandas as pd
import json
from collections import defaultdict

# 1. Setup spaCy
# Using 'disable' speeds up processing by only running the Tagger
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def analyze_benchmark(name, prompts):
    """Parses prompts and returns counts for Nouns, Adjectives, and Verbs."""
    unique_nouns = set()
    unique_adjs = set()
    unique_verbs = set()
    total_words = 0

    print(f"Processing {name}...")
    for doc in nlp.pipe(prompts, batch_size=256):
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            
            total_words += 1
            pos = token.pos_
            lemma = token.lemma_.lower()
            
            if pos == "NOUN" or pos == "PROPN": # Including Proper Nouns for benchmarks
                unique_nouns.add(lemma)
            elif pos == "ADJ":
                unique_adjs.add(lemma)
            elif pos == "VERB":
                unique_verbs.add(lemma)

    return {
        "Benchmark": name,
        "Total Prompts": len(prompts),
        "Total Words": total_words,
        "Unique Nouns": len(unique_nouns),
        "Unique Adjs": len(unique_adjs),
        "Unique Verbs": len(unique_verbs)
    }

# 2. Load your data (Replace paths with your actual local paths)
# --- EvalMuse ---
# Assuming evalmuse_df is your loaded dataframe
# evalmuse_prompts = evalmuse_df['caption'].tolist()

# --- DPGBench ---
# dpg_prompts = [json.loads(line)['prompt'] for line in open('dpgbench.jsonl')]

# --- GenEval ---
# geneval_prompts = [json.loads(line)['prompt'] for line in open('geneval.jsonl')]

# 3. Execution & Comparison
# (Dummy list for demonstration; replace with actual loaded lists)
benchmarks = {
    "EvalMuse": evalmuse_prompts, 
    "DPGBench": dpg_prompts, 
    "GenEval": geneval_prompts
}

results = []
for name, prompts in benchmarks.items():
    results.append(analyze_benchmark(name, prompts))

# 4. Display Results
df_results = pd.DataFrame(results)
print("\n--- POS Analysis Comparison ---")
print(df_results.to_string(index=False))