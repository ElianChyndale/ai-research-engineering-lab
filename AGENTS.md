# AGENTS.md — AI Research Engineering Lab

## Repository Purpose

Reusable environment for learning ML/IR foundations, running deterministic experiments,
recording configurations and results, and practising baseline/ablation design.

## Allowed File Scope

- `src/airelab/` — library code
- `configs/` — experiment YAML configs
- `scripts/` — CLI entry points
- `tests/` — test suite
- `docs/` — templates and documentation
- `research/` — generated results and logs (fixtures only unless reviewed)
- `.github/workflows/` — CI

Do NOT modify sibling repositories (ecoquant-financial-intelligence,
green-bond-market-infrastructure) or coordination/SOL contracts.

## Test-First Requirement

Write a failing test before implementing a feature. Every PR must pass:
- `python -m pytest -q`
- `python -m mypy src/`
- `python -m ruff check src/ tests/`

## Deterministic Experiment Requirement

Every experiment must:
- Accept a `seed` parameter
- Produce identical results given identical config and seed
- Write a manifest with SHA-256 artifact hashes
- Use only local synthetic fixture data (no network calls)

## No Unsupported Research Claims

Results under `research/` are educational fixtures unless a human has reviewed
and promoted them. Do not claim production readiness, statistical significance
from single runs, or superiority over established libraries.

## No Secrets

No API keys, tokens, or credentials in any file. No `.env` files.

## No Large Model Downloads

No model weights, embeddings, or large datasets. All data is synthetic and
generated at experiment time.

## Fixture vs Research Result

- `tests/fixtures/` — small deterministic inputs for unit tests
- `research/results/` — generated experiment outputs (educational unless reviewed)
- A fixture run CANNOT be labelled "reviewed research"
