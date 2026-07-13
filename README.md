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
| `airelab.core`       | Config, seeds, artifacts, manifest, validation               |
| `airelab.foundations`| Linear regression, logistic regression, PCA, BM25, calibration |
| `airelab.experiments`| Registry, runner, comparison                                 |

## Running Experiments

```bash
python scripts/run_experiment.py --config configs/linear_regression.yaml
python scripts/run_experiment.py --config configs/logistic_regression.yaml
python scripts/run_experiment.py --config configs/bm25.yaml
python scripts/validate_artifacts.py --run-dir research/results/<run-id>
```

## Limitations

- All data is synthetic; no real-world datasets
- No heavy ML frameworks (scikit-learn, PyTorch, TensorFlow)
- Single-run experiments; no statistical significance claims
- Educational implementations, not production-grade
- Results are fixtures unless human-reviewed

## Repository Structure

```
src/airelab/       — library code
configs/           — experiment YAML configs
scripts/           — CLI entry points
tests/             — unit and integration tests
research/          — generated results and logs
docs/              — templates and documentation
```

## Relationship to Other Repos

This lab supports learning and does not replace:
- **EcoQuant** — financial intelligence platform
- **Green Bond Market Infrastructure** — bond market systems
