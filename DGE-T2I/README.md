# DGE-T2I

`DGE-T2I` contains the text-to-image portion of the DGE repository: benchmark data, image-generation utilities, DGE-FineEval, ablation experiments, cached result summaries, and visualization utilities.

## What This Directory Provides

- **Benchmark inputs:** prompt and scene-graph files under `data/raw/`.
- **DGE-FineEval:** object grounding, attribute binding, and relation verification pipelines under `src/eval/`.
- **Ablations:** stage-replacement experiments comparing VLM and specialist model components.
- **Generation utilities:** per-model image generation scripts under `src/models/` and `scripts/`.
- **Cached outputs:** small summaries and paper-result artifacts under `data/raw/eval_v1/` and `reports/`.

Generated images, embeddings, model checkpoints, and local model snapshots are not release assets in git. They should be downloaded or regenerated using the artifact manifests in `../artifacts/manifest/`.

## Install

```bash
cd DGE-T2I
python -m pip install -e .
```

The base package metadata is in `pyproject.toml`. GPU/full-evaluation environments also need model-specific dependencies such as vLLM, Qwen-VL compatible checkpoints, ReITR/RelTR, EVA-CLIP, BLIP-2, and generation-model dependencies.

## Lightweight Checks

```bash
python test_modular_structure.py
```

`test_graph_ablation.py` is intended to validate the ablation harness without loading large models. If it fails on import, fix the package import path before using the ablation code in a clean environment.

## Cached Reproduction

Cached reproduction uses existing summaries to inspect and rebuild paper-level tables without rerunning model inference.

Relevant inputs:

- `data/raw/eval_v1/*_eval_summary.json`
- `data/raw/eval_v1/eval_summary_accuracies.csv`
- `data/images/survey_samples/pair_preferences.csv`
- `reports/pair_metrics*.json`
- `reports/ablation/*.json`
- `reports/baselines/*.json` and `*.csv`

See `../docs/paper_results_map.md` for how these files map to paper tables and figures.

## Full Rerun

A full rerun has three phases:

1. Generate or download the target model images into `data/images/<model>/`.
2. Run DGE-FineEval against the benchmark scene graphs.
3. Run ablation permutations and aggregate paper summaries.

Current scripts are working research launchers and may contain cluster assumptions. Before a full rerun, set paths through environment variables or config:

- `QWEN_MODEL_PATH`
- `VLLM_API_BASE`
- `REITR_CODE_DIR`
- `REITR_CHECKPOINT_PATH`
- `EVA_CLIP_CODE_DIR`
- `BLIP2_MODEL_PATH`
- `CUDA_VISIBLE_DEVICES`

Do not commit generated outputs from a rerun. Write new runs to ignored local directories and promote only small, documented summaries when needed.

## Data and Artifact Policy

Tracked data should stay limited to benchmark definitions, small summaries, and documentation. These paths are intentionally ignored for local/generated assets:

- `data/models/`
- `data/images/`
- `data/embeddings/`
- `runs/`
- `logs/`
- `offload/`

Use the manifests under `../artifacts/manifest/` as the source of truth for any external artifact required by cached or full reproduction.
