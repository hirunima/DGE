# Artifact Policy

The DGE repository keeps source code, benchmark definitions, documentation, and small paper summaries in git. Large or machine-specific assets are externalized.

## Externalized Assets

Do not commit:

- model checkpoints and Hugging Face snapshots
- generated images
- prompt embeddings
- vLLM containers or local model caches
- scratch run directories
- local cluster logs

Expected external artifacts are described by JSON manifests under `artifacts/manifest/`.

## Manifest Fields

Each manifest entry should include:

- `name`: stable artifact name
- `category`: `benchmark`, `model`, `generated-images`, `embedding`, or `results`
- `required_for`: `cached`, `full`, or `optional`
- `expected_path`: path relative to the repository root after download or generation
- `status`: `tracked`, `external`, or `regenerate`
- `notes`: short instructions or source information

When publishing an external artifact, add `size_bytes` and `sha256`.

## Current Policy

For the cleanup release:

- Keep benchmark JSON/JSONL files and small summaries in git.
- Keep generated images, embeddings, and model snapshots outside git.
- Keep the historical `DGE-T2I-og/` tree outside the release workflow.
- Store future full-run outputs under ignored run directories until intentionally summarized.

