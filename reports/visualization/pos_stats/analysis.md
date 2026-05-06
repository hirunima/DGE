# Prompt Complexity Analysis

All datasets are measured from their prompt text with spaCy. If `en_core_web_sm` is unavailable, the script uses a lightweight lexical fallback and records that in `summary.csv`.

## Where Ours Stands

- Attribute binding is strong but not the top under spaCy-only scoring: ours has 0.48 attributes/object; the strongest baseline is TIIF-Bench (long) at 0.55.
- Attribute vocabulary is mid-to-high: ours has 629 unique spaCy adjective lemmas; the largest comparison is TIIF-Bench (long) with 2207.
- Relation vocabulary is broad for a moderate-length prompt set: ours has 534 unique spaCy relation-term lemmas; the largest comparison is TIIF-Bench (long) with 1769.
- Relations/prompt are competitive without relying on very long prose: ours averages 5.52, higher than 6 of 9 comparison datasets.
- Object coverage stays multi-object: ours averages 6.39 objects/prompt, higher than 6 of 9 comparison datasets.

## Distribution Checks

- Objects/Prompt: Ours averages 6.39. It is higher than 6 of 9 comparison datasets; the largest prompt-estimate baseline is TIIF-Bench (long) at 26.71.
- Attributes/Prompt: Ours averages 2.92. It is higher than 6 of 9 comparison datasets; the largest prompt-estimate baseline is TIIF-Bench (long) at 13.78.
- Attributes/Object: Ours averages 0.48. It is higher than 7 of 9 comparison datasets; the largest prompt-estimate baseline is TIIF-Bench (long) at 0.55.
- Relations/Prompt: Ours averages 5.52. It is higher than 6 of 9 comparison datasets; the largest prompt-estimate baseline is TIIF-Bench (long) at 22.11.

## Ours At A Glance

- Prompts: 3099
- Unique objects: 2000
- Unique attributes: 629
- Unique relations: 534
- Mean objects/prompt: 6.39
- Mean attributes/prompt: 2.92
- Mean attributes/object: 0.48
- Entities with attributes: 7485 (37.8%)
- Mean relations/prompt: 5.52
- Mean relations/object: 0.86

## Interpretation

Under spaCy-only scoring, the longest descriptive datasets lead on raw object, attribute, and relation counts. Ours remains a moderate-length, multi-object benchmark with competitive attribute density and relation density, so it is useful for testing compositional grounding without relying on very long prompt descriptions.

See `summary.csv`, `per_prompt_distributions.csv`, `top_terms.csv`, and the PNG plots in this directory for the underlying numbers.
