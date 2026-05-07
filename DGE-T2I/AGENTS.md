# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Python package code.
- `src/data/` holds data-generation entry points; modular code lives in `src/data/modules/` (config, processing, model, pipeline).
- `src/eval/`, `src/models/`, and `src/visualization/` contain evaluation, model runners, and analysis utilities.
- `data/` stores datasets and outputs; generated artifacts land in `data/raw/`.
- `test_input/` provides sample inputs for lightweight runs; `test_*.py` are standalone test scripts.
- `scripts/` includes job scripts (e.g., `scripts/generate_data.sh` for GPU/HPC runs).

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode from `pyproject.toml`.
- `python -m src.data.generate --samples 2` runs data generation locally (see `scripts/generate_data.sh` for cluster/HPC setup).
- `python test_modular_structure.py` verifies imports and module wiring without running the full model.
- `python test_processing.py` processes `test_input/` files to validate data flow.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8 style.
- Use `snake_case` for functions/variables and `PascalCase` for classes.
- Keep module boundaries clear: data pipeline logic goes under `src/data/modules/`.
- No formatter or linter is configured; keep imports tidy and avoid unused symbols.

## Testing Guidelines
- Tests are lightweight scripts (not pytest). Name them `test_*.py` and keep them runnable via `python`.
- Prefer tests that avoid GPU-only dependencies unless explicitly required.
- When adding modules, include an import/structure check similar to `test_modular_structure.py`.

## Commit & Pull Request Guidelines
- Recent commits use short, sentence-case, past-tense messages (e.g., “Updated code for …”, “Created new set of prompts”).
- PRs should include: purpose, key changes, how to reproduce, and any data/output artifacts affected.
- If you add or regenerate data, describe the source inputs and update paths under `data/raw/`.

## Environment & Data Notes
- The generation script assumes a CUDA-capable environment and, in `scripts/generate_data.sh`, a `conda` env named `t2i-r1` plus module loads.
- Avoid committing large generated outputs unless explicitly requested; prefer documenting how to reproduce them.
- If pre-existing changes are detected after edits, stop and ask whether they were intended before proceeding.
