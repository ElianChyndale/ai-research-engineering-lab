# AI Research Engineering Lab

**This is a research-engineering and foundations laboratory. It is not a production AI system.**

It provides deterministic experiments and learning implementations for ML/IR foundations.
Results are educational fixtures unless explicitly reviewed.

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Current Modules

| Module               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `airelab.core`       | Config, seeds, artifacts, manifest, validation, aggregation  |
| `airelab.foundations`| Linear regression, logistic regression, PCA, BM25, calibration |
| `airelab.experiments`| Registry, runner, comparison, multi-seed, repeated eval, cross-validation |

## Running Experiments

### Single Deterministic Run

```bash
python scripts/run_experiment.py --config configs/linear_regression.yaml
python scripts/run_experiment.py --config configs/logistic_regression.yaml
python scripts/run_experiment.py --config configs/bm25.yaml
python scripts/run_experiment.py --config configs/pca.yaml
python scripts/validate_artifacts.py --run-dir research/results/<run-id>
```

### Multi-Seed Variance

```bash
python scripts/run_multi_seed.py --config configs/multi_seed_lr.yaml
python scripts/run_multi_seed.py --config configs/multi_seed_logr.yaml
python scripts/run_multi_seed.py --config configs/multi_seed_pca.yaml
```

### Repeated Train/Test Evaluation

```bash
python scripts/run_repeated_eval.py --config configs/repeated_eval_lr.yaml
python scripts/run_repeated_eval.py --config configs/repeated_eval_logr.yaml
```

### Cross-Validation

```bash
python scripts/run_cross_validation.py --config configs/cv_lr.yaml
python scripts/run_cross_validation.py --config configs/cv_logr.yaml
```

### Comparing Experiment Families

```bash
python scripts/compare_families.py research/results/lr-multiseed-001 research/results/logr-multiseed-001
```

## Teaching Progression

This lab teaches the distinction between:

1. **One deterministic run** — single seed, single split, one set of metrics
2. **Multi-seed variance** — same experiment, different seeds, observe metric spread
3. **Repeated evaluation** — different train/test splits, observe generalization variance
4. **Cross-validation** — k-fold structured splits, aggregate train/test performance
5. **Final held-out testing** — reserved data never used during development

## Limitations

- All data is synthetic; no real-world datasets
- No heavy ML frameworks in core (`src/airelab/`)
- Educational implementations, not production-grade
- Results are fixtures unless human-reviewed
- No statistical significance claims from single experiments

## Repository Structure

```
src/airelab/       — library code
configs/           — experiment YAML configs
scripts/           — CLI entry points
tests/             — unit and integration tests
research/          — generated results and logs
docs/              — templates and documentation
labs/              — optional extended labs (requires requirements.txt)
```

## Relationship to Other Repos

This lab supports learning and does not replace:
- **EcoQuant** — financial intelligence platform
- **Green Bond Market Infrastructure** — bond market systems
