# Paper Results Map

This file maps paper-level text-to-image results to repository files.

## DGE-FineEval T2I Model Scores

Primary cached summaries:

- `DGE-T2I/data/raw/eval_v1/eval_summary_accuracies.csv`
- `DGE-T2I/data/raw/eval_v1/flux_eval_summary.json`
- `DGE-T2I/data/raw/eval_v1/qwen-image_eval_summary.json`
- `DGE-T2I/data/raw/eval_v1/sd15_eval_summary.json`
- `DGE-T2I/data/raw/eval_v1/sdxl_eval_summary.json`
- `DGE-T2I/data/raw/eval_v1/z-image_eval_summary.json`

Detailed evaluator outputs are the matching `*_eval_results.json` files in the same directory.

## Human Preference Correlation

Primary cached inputs:

- `DGE-T2I/data/images/survey_samples/pair_preferences.csv`
- `DGE-T2I/reports/pair_metrics_v1.json`
- `DGE-T2I/reports/pair_metrics_ablation*.json`
- `DGE-T2I/reports/baselines/pair_metrics_ablation_baselines.json`
- `DGE-T2I/reports/baselines/pair_preferences_ablation_baselines.csv`

These files support paper comparisons among DGE-FineEval, DSG/VQA-style scores, PickScore/ImageReward-style baselines, and human pairwise preferences.

## Metric Ablation Experiments

Primary cached outputs:

- `DGE-T2I/reports/ablation/aggregate_matrix.json`
- `DGE-T2I/reports/ablation/aggregate_matrix.csv`
- `DGE-T2I/reports/ablation/latency_report.json`
- `DGE-T2I/reports/ablation/relation_swap_report.json`
- `DGE-T2I/reports/ablation/run_metadata.json`

Detailed permutation files live under `DGE-T2I/reports/ablation/**/permutations_*`.

## Qualitative Figures

Prompt grids and qualitative examples are under:

- `DGE-T2I/reports/prompt_grids/`
- `DGE-T2I/reports/baselines/v1_v2_v3_beats_baselines_cases/`
- `DGE-T2I/reports/visualization/`

These are derived visualization artifacts. Regenerate them from cached summaries and downloaded/generated images when possible.

