# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Undertale is a research project for training LLMs on binary program
understanding. It includes dataset build pipelines, custom transformer models,
evaluation infrastructure, and integrations with external products.

## Architecture

- `undertale/` — installable library (schema, parsers, pipeline utilities, models)
  - `pipeline/` — per-format stage helpers (binary, parquet, dask, json, tarfile, zip, cpp)
  - `models/` — transformer variants (maskedlm, classification, summarization, tuning, density)
  - `schema.py` — Pandera DataFrame schemas; all pipeline outputs must validate against one
  - `parsers.py` — `DatasetArgumentParser` / `ModelArgumentParser` for pipeline entry points
- `pipelines/` — standalone runnable scripts (not part of the library)
  - `datasets/` — data ingestion pipelines
  - `models/` — training, fine-tuning, inference scripts; paired `.slurm` files for HPC runs
- `extras/` — standalone sub-projects (inference-frontend, inference-server)
- `scripts/` — one-off migration scripts (datatrove-legacy)
- `environments/` — `.env` templates; source one before running pipelines

### Notes

#### Pipeline Stages
- **Pipeline Idempotency**: stages use `get_or_create_directory(output)` which
  returns `(path, created)`. If `not created`, the stage should skip processing
  and return existing outputs. All new pipeline stages must follow this
  pattern.

## Commands

```bash
# Run all tests
python tests/unit.py --verbose

# Run a specific test case
python tests/unit.py --verbose TestPipelineBinary

# Pre-commit hooks (linting/formatting).
pre-commit run -a

# Build documentation
sphinx-build -b html docs build/documentation/
```

## Workflow

Always run `pre-commit run -a` after any code changes before reporting a task
complete. Hooks auto-fix formatting (black, isort).

## Conventions

- **Code style**: Black formatter, isort with `profile = "black"`.
- **Docstrings**: Google style.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, etc.
- **Branching**: Feature branches only; squash-merge to `main`.
- **Type hints**: Used throughout; mypy is enforced in CI.

## Style

- Prefer variable/function/class names that are complete words (e.g.,
  `position` instead of `pos`).
- Light preference against multi-word variable/function/class names. Ignore
  this preference if using a single word would result in too much ambiguity and
  make reading code difficult.
