# Reproducibility Guide

This guide describes how to reproduce the DGE-T2I results in two modes: cached reproduction and full rerun.

## 1. Environment

Start with the lightweight package install:

```bash
cd DGE-T2I
python -m pip install -e .
python test_modular_structure.py
```

The lightweight test validates the repository structure without loading VLMs or generation models.

Full reruns require CUDA-capable GPUs and access to the model checkpoints used by the paper. At minimum, expect to configure:

- Qwen3-VL or compatible VLM evaluator
- vLLM endpoint for VLM-backed stages, when using `--use-vllm`
- ReITR/RelTR code and checkpoint for specialist relation scoring
- EVA-CLIP or BLIP-2 for specialist attribute scoring
- Text-to-image model checkpoints for image generation

## 2. Cached Reproduction

Use cached reproduction when the goal is to inspect paper outputs or rebuild tables from existing summaries. Inputs are already present as small JSON/CSV files:

- `DGE-T2I/data/raw/eval_v1/`
- `DGE-T2I/reports/`
- `DGE-T2I/data/images/survey_samples/pair_preferences.csv`

Large image and model artifacts are externalized. See `artifacts/manifest/paper_results.json` for the cached-result manifest.

## 3. Full T2I Rerun

The full T2I rerun consists of:

1. Prepare prompt/scene-graph inputs from `DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json`.
2. Generate five images per prompt for each target model into `DGE-T2I/data/images/<model>/`.
3. Run DGE-FineEval over generated images and scene graphs.
4. Aggregate summaries into `*_eval_summary.json` and paper-level CSV files.

The checked-in launch scripts are research scripts. Before a full rerun, replace machine-specific defaults with environment variables:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export VLLM_API_BASE=http://127.0.0.1:8000/v1
export REITR_CODE_DIR=/path/to/RelTR
export REITR_CHECKPOINT_PATH=/path/to/checkpoint0149.pth
export BLIP2_MODEL_PATH=Salesforce/blip2-itm-vit-g
```

## 4. Ablation Rerun

The ablation pipeline compares stage choices:

- `E1`: GroundingDINO + CLIP object grounding
- `V1`: Qwen3-VL object grounding
- `E2`: EVA-CLIP or BLIP-2 attribute scoring
- `V2`: Qwen3-VL/MolmoPoint attribute scoring
- `S2`: skip stage 2
- `E3`: ReITR relation scoring
- `V3`: Qwen3-VL relation scoring

The paper ablation result of interest is the permutation matrix comparing all combinations of E/V stage choices, including `V1-V2-V3`.

## 5. Output Hygiene

Write local reruns to ignored directories such as `DGE-T2I/runs/` or `DGE-T2I/reports/local/`. Promote only small summaries to tracked result directories, and document any promoted file in the artifact/result map.

