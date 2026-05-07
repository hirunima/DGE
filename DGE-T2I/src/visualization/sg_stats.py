import sng_parser
import pandas as pd
import json
from collections import defaultdict
import numpy as np

# 1. Setup spaCy
# # Keep the parser enabled for adjective-to-noun dependency links.
# nlp = spacy.load("en_core_web_sm", disable=["ner"])

def analyze_benchmark(name, prompts):
    """Parses prompts and returns counts for Nouns, Adjectives, and Verbs."""
    sgs = []
    for prompt in prompts: 
        sgs.append(sng_parser.parse(prompt))

    total_prompts = len(sgs)
    entity_counts = []
    relation_counts = []
    modifier_counts = []
    entities_with_modifiers = 0
    unique_entity_heads = set()
    unique_relation_labels = set()

    for sg in sgs:
        entities = sg.get("entities", [])
        relations = sg.get("relations", [])
        entity_counts.append(len(entities))
        relation_counts.append(len(relations))

        for entity in entities:
            head = entity.get("lemma_head") or entity.get("head")
            if head:
                unique_entity_heads.add(head)
            modifiers = entity.get("modifiers", [])
            modifier_counts.append(len(modifiers))
            if modifiers:
                entities_with_modifiers += 1

        for relation in relations:
            label = relation.get("relation")
            if label:
                unique_relation_labels.add(label)

    total_entities = sum(entity_counts)
    total_relations = sum(relation_counts)
    total_modifiers = sum(modifier_counts)
    avg_entities_per_prompt = total_entities / total_prompts if total_prompts else 0
    avg_relations_per_prompt = total_relations / total_prompts if total_prompts else 0
    avg_modifiers_per_entity = total_modifiers / total_entities if total_entities else 0
    pct_entities_with_modifiers = (entities_with_modifiers / total_entities * 100) if total_entities else 0
    avg_relations_per_entity = total_relations / total_entities if total_entities else 0

    return {
        "Benchmark": name,
        "Prompts": total_prompts,
        "Total Entities": total_entities,
        "Total Relations": total_relations,
        "Avg Entities/Prompt": avg_entities_per_prompt,
        "Avg Relations/Prompt": avg_relations_per_prompt,
        "Avg Modifiers/Entity": avg_modifiers_per_entity,
        "Pct Entities w/Modifiers": pct_entities_with_modifiers,
        "Avg Relations/Entity": avg_relations_per_entity,
        "Unique Entity Heads": len(unique_entity_heads),
        "Unique Relation Labels": len(unique_relation_labels),
    }

# 2. Load your data (Replace paths with your actual local paths)
# --- EvalMuse ---
# Assuming evalmuse_df is your loaded dataframe
# evalmuse_prompts = evalmuse_df['caption'].tolist()

# --- TIIF-Bench ---
tiif_bench_prompts = [json.loads(line)['short_description'] for line in open('../../TIIF-Bench/data/test_prompts/all_prompts.jsonl')]

tiif_bench_prompts_l = [json.loads(line)['long_description'] for line in open('../../TIIF-Bench/data/test_prompts/all_prompts.jsonl')]


# --- DPGBench ---
dpg_prompts = [json.loads(line)['prompt'] for line in open('../../ELLA/dpg_bench/prompts.jsonl')]

# --- GenEval ---
# geneval_prompts = [json.loads(line)['prompt'] for line in open('geneval.jsonl')]

our_prompts = [obj['prompt'] for obj in json.load(open('data/raw/qwen8b_t2i_prompts_aug_v1.json'))]

# --- T2I-CompBench++ ---
# https://huggingface.co/datasets/NinaKarine/t2i-compbench
# Parquet files per category, each with a "text" column; concatenate all val splits.
import glob as _glob
_t2i_files = _glob.glob('../../T2I-CompBench/**/*val*.parquet', recursive=True)
t2i_compbench_prompts = pd.concat([pd.read_parquet(f) for f in _t2i_files])['text'].dropna().unique().tolist()

# --- GenAI-Bench ---
# https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600
# JSON dict keyed by id, each entry has a "prompt" field.
genai_bench_prompts = [v['prompt'] for v in json.load(open('../../GenAI-Bench/genai_image.json')).values()]

# --- GECKONUM ---
# https://github.com/google-deepmind/geckonum_benchmark_t2i
# CSV with a "prompt" column.
geckonum_prompts = pd.read_csv('../../GECKONUM/prompts.csv')['prompt'].dropna().tolist()

# 3. Execution & Comparison
# (Dummy list for demonstration; replace with actual loaded lists)
benchmarks = {
    "TIIF-Bench": tiif_bench_prompts,
    "TIIF-Bench (long)": tiif_bench_prompts_l,
    "DPGBench": dpg_prompts,
    # "GenEval": geneval_prompts,
    "T2I-CompBench++": t2i_compbench_prompts,
    "GenAI-Bench": genai_bench_prompts,
    "GECKONUM": geckonum_prompts,
    "Ours": our_prompts,
}

results = []
for name, prompts in benchmarks.items():
    results.append(analyze_benchmark(name, prompts))

# 4. Display Results
df_results = pd.DataFrame(results)
print("\n--- POS Analysis Comparison ---")
print(df_results.to_string(index=False))
