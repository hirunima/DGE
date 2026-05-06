# DGE
DGE - A Dynamic Metric and Grounded Evaluation Benchmarks for Text-to-Image and Image Editing Models

## Overview

Recent advancements in text-to-image (T2I) generation and image editing models have produced highly realistic visual results. However, evaluating these models remains a major challenge. Existing evaluation metrics often rely on global image statistics, high-level semantic similarity, or fixed VLM-based scoring criteria, which fail to capture fine-grained compositional details, object relationships, and attribute grounding specified in prompts.

This repository provides a generative evaluation framework featuring:

- Grounded benchmarks for text-to-image generation and image editing
- Fine-grained scene-graph-based evaluation
- Global scene-graph-conditioned consistency evaluation

---

## Abstract

Recent advancements in text-to-image (T2I) generation and image editing models have produced stunning visual results. However, evaluating these models remains a significant challenge. Existing metrics often rely on global image statistics, high-level semantic similarity (e.g., CLIPScore), or VLM-based scoring that applies a fixed set of criteria to all samples. These approaches broadly fail to capture the fine-grained compositional details, object relationships, and attribute grounding specified in text prompts.

We propose a generative evaluation framework featuring two large-scale grounded benchmarks and two novel dynamic metrics. For text-to-image synthesis, we introduce a diverse `(text, scene graph)` pair benchmark to enable granular compositional assessment. For image editing, we create a benchmark of `(source graph, target graph, prompt)` designed to facilitate precise, instruction-based edits.

Using these benchmarks as a foundation, we introduce two novel evaluation metrics for systematic assessment. Our VLM and scene-graph-based fine-grained metric can assess object-level, attribute-level, and relationship-level alignment. Consequently, we propose a global scene-graph-conditioned consistency metric that measures semantic alignment in a shared embedding space, capturing global compositional structure and semantic consistency across datasets.

Our experiments demonstrate that the combined framework provides a more holistic, accurate, and human-aligned assessment of state-of-the-art models, revealing limitations not captured by previous metrics.

---

# Repository Structure

```bash
.
├── prompt_generation/
│   ├── text_to_image/
│   ├── image_editing/
│   └── scene_graphs/
│
├── evaluation/
│   ├── fine_grain/
│   │   ├── object_level/
│   │   ├── attribute_level/
│   │   ├── relationship_level/
│   │   └── vlm_scoring/
│   │
│   └── global/
│       ├── graph_embeddings/
│       ├── semantic_alignment/
│       └── consistency_metrics/
│
├── benchmarks/
│   ├── t2i/
│   └── image_editing/
│
├── scripts/
├── configs/
└── README.md
```

---

# Prompt Generation

The `prompt_generation/` directory contains code for generating prompts, grounded scene graphs, and benchmark annotations for:

- Text-to-image generation
- Image editing

---

# Evaluation

The `evaluation/` directory contains two evaluation pipelines:

## Fine-Grain Evaluation

Located in:

```bash
evaluation/fine_grain/
```

Includes evaluation for:

- Object-level alignment
- Attribute-level alignment
- Relationship-level alignment
- VLM-based scene-graph verification

---

## Global Evaluation

Located in:

```bash
evaluation/global/
```

Includes evaluation for:

- Scene-graph-conditioned embedding alignment
- Global semantic consistency
- Cross-dataset compositional evaluation

---

# Benchmarks

## Text-to-Image Benchmark

Benchmark format:

```text
(text prompt, scene graph)
```

---

## Image Editing Benchmark

Benchmark format:

```text
(source graph, target graph, edit prompt)
```

---

# Usage

## Prompt Generation

```bash
python scripts/generate_prompts.py
```

---

## Fine-Grain Evaluation

```bash
python scripts/run_fine_grain_eval.py
```

---

## Global Evaluation

```bash
python scripts/run_global_eval.py
```

---

# Citation

```bibtex
@article{yourpaper2026,
  title={Fine-Grained and Global Evaluation Framework for Text-to-Image and Image Editing Models},
  author={Anonymous},
  year={2026}
}
```

---

# License

MIT License
