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
| `airelab.core`       | Config, seeds, artifacts, manifest, validation, aggregation, **statistics, provenance, lifecycle, kill_gates, leakage** |
| `airelab.foundations`| Linear regression, logistic regression, PCA, BM25, calibration |
| `airelab.experiments`| Registry, runner, comparison, multi-seed, repeated eval, cross-validation |
| `airelab.cheap_kill` | **ResearchHypothesis / ExperimentSpec / KillTestReport schemas (direction-agnostic cheap-kill interface)** |
| `airelab.torch`      | **PyTorch research core: reproducibility, checkpoint, metrics, gradient_check, trainer** |

## Research Harness (for cheap-kill falsification experiments)

Added 2026-08-07 to make future Top-3 falsification experiments 2-5x faster and
more reproducible. Direction-agnostic (no DFL/belief/finance/graph anchoring):

- `airelab.core.statistics` — bootstrap CI, paired differences (seed-aligned),
  Cohen's d, practical effect-margin (`beats_by_margin`).
- `airelab.core.provenance` — git head / dirty / config hash / python / deps /
  timestamp seal (the "IPW moment" provenance discipline).
- `airelab.core.lifecycle` — `FrozenConfig` (EXPLORATORY→FROZEN→CONFIRMATORY→
  INVALIDATED); detects config modification before confirmatory promotion.
- `airelab.core.kill_gates` — pre-registered PASS/WARNING/FAIL/KILL gates saved
  before analysis.
- `airelab.core.leakage` — learner-view vs evaluator-view separation (structural
  AST scan + runtime guard).
- `airelab.cheap_kill.schemas` — `ResearchHypothesis`, `ExperimentSpec`,
  `KillTestReport` (fill in per candidate, instantiate in <1 day).
- `airelab.torch` — deterministic seed control, device handling, checkpointing,
  early stopping, gradient-norm logging, finite-difference gradient check,
  ECE calibration, a shallow trainer, and seed-disjointness utilities.

`reproductions/template/` forces the pre-coding disciplines (paper claim, owner,
IPW moment, strongest baseline, kill condition) before any code.

## Reproductions

- `reproductions/template/` — the reproduction skeleton (paper_claim, baseline_map,
  config, run, analyse, report).

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
- Standard deviation uses sample std (ddof=1)
- Repeated evaluation variance combines data-generation, split, and model-training variance (not pure split variance)
- Cross-validation is for model assessment, not a final held-out test estimate

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
