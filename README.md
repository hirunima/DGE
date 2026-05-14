# DGE

DGE is the official repository for **DGE: A Dynamic Metric and Grounded Evaluation Benchmarks for Text-to-Image and Image Editing Models**.

The current release focuses on making the text-to-image evaluation path reproducible. The `DGE-T2I` directory contains the implementation for DGE-FineEval, the model-generation utilities, cached evaluation outputs, and ablation experiments used for the paper's text-to-image results.

## Repository Layout

```text
.
├── DGE-T2I/                  # DGE-FineEval and T2I ablation implementation
├── docs/                     # Reproducibility and artifact documentation
├── artifacts/manifest/       # External artifact manifests and checksums
├── dataset_pipeline.pdf      # Benchmark construction figure
└── 526_DGE_A_Dynamic_Metric_and_G.pdf
```

`DGE-T2I-og/`, when present locally, is a historical working copy and is not part of the release workflow.

## Reproduction Modes

The repository supports two reproduction modes:

- **Cached reproduction:** use checked-in summaries plus external artifacts to regenerate paper tables and inspect result files without rerunning large models.
- **Full rerun:** regenerate images and rerun DGE-FineEval/ablation pipelines with the required GPUs, checkpoints, and VLM services.

Large assets are intentionally kept outside git. This includes generated images, prompt embeddings, model checkpoints, local Hugging Face snapshots, and scratch run outputs. See [docs/artifacts.md](docs/artifacts.md) for the artifact policy and manifests.

## Quickstart

```bash
cd DGE-T2I
python -m pip install -e .
python test_modular_structure.py
```

The lightweight checks do not require GPU inference. Full evaluation requires CUDA-capable hardware and access to the VLM/generation models described in [docs/reproducibility.md](docs/reproducibility.md).

## Paper Result Map

The primary paper result mapping is documented in [docs/paper_results_map.md](docs/paper_results_map.md). In short:

- DGE-FineEval T2I summaries live under `DGE-T2I/data/raw/eval_v1/`.
- Human preference and pairwise metric summaries live under `DGE-T2I/data/images/survey_samples/` and `DGE-T2I/reports/`.
- Ablation summaries live under `DGE-T2I/reports/ablation/` and `DGE-T2I/reports/baselines/`.

## Citation

```bibtex
@article{dge2026,
  title={DGE: A Dynamic Metric and Grounded Evaluation Benchmarks for Text-to-Image and Image Editing Models},
  author={Anonymous},
  year={2026}
}
```
