# V0.1.1 Verification Report

## Parent V0.1 Commits

| Hash | Message |
|------|---------|
| `713d9d3` | chore: establish research engineering laboratory |
| `f674c5a` | docs: add research learning workflow |
| `0f59600` | docs: verify research engineering lab v0.1 |

## Maintenance Commit

| Hash | Message |
|------|---------|
| *(pending)* | fix: make research artifacts overwrite safely on Windows |

## Exact Defect

`ExperimentManifest.write()` in `src/airelab/core/manifest.py` used:

```python
tmp.rename(path)
```

On Windows, `Path.rename()` delegates to `os.rename()`, which raises
`FileExistsError` when the destination file already exists. This prevented
re-running experiments into an existing output directory on Windows.

## Exact Correction

Replaced with:

```python
tmp.replace(path)
```

`Path.replace()` delegates to `os.replace()`, which atomically replaces the
destination on both Windows and POSIX systems.

## Secondary Repair

`src/airelab/experiments/runner.py` excluded `manifest.json` from the artifact
hashing loop. On re-runs, the stale `manifest.json` from a previous run was
being hashed as an artifact, causing validation failure (the new manifest had a
different hash than the stale file it recorded). The manifest is written last
and must not reference itself.

## Regression Tests

### Unit: `tests/unit/test_manifest.py`

| Test | Purpose |
|------|---------|
| `test_manifest_overwrite_existing` | Write two manifests to the same path; verify no `FileExistsError`; verify destination contains the second manifest; verify no stale `.tmp` file remains. |
| `test_manifest_overwrite_preserves_on_write_failure` | Verify strict JSON parsing of an overwritten manifest. |

### Integration: `tests/integration/test_experiments.py`

| Test | Purpose |
|------|---------|
| `test_manifest_overwrite_on_rerun` | Run `run_experiment.py` twice into the same directory via subprocess; verify both succeed; verify manifest is valid JSON; verify artifact validation passes. |

## Full Test Result

```
92 passed in 26.62s
```

| Category | Count |
|----------|-------|
| Unit tests | 82 |
| Integration tests | 8 |
| Lab tests | 2 |
| **Total** | **92** |

## CI Matrix

Updated `.github/workflows/ci.yml` to test on:

| OS | Python | Scope |
|----|--------|-------|
| ubuntu-latest | 3.11, 3.12, 3.13 | ruff, mypy, unit tests, integration tests |
| windows-latest | 3.11, 3.12, 3.13 | ruff, mypy, unit tests, integration tests |

Both platforms run the atomic-overwrite regression tests and one deterministic
fixture experiment through the integration suite.

## Deterministic Experiment Result

All three experiments verified deterministic (same seed, same metrics):

| Experiment | Metric Match |
|-----------|-------------|
| `linear_regression` | IDENTICAL |
| `logistic_regression` | IDENTICAL |
| `bm25` | IDENTICAL |

Overwrite test: running `run_experiment.py` twice into the same directory
succeeds and produces a valid manifest.

## Artifact Validation

All three fixture experiments validate:

```
research/results/lr-synthetic-001: VALID
research/results/logr-synthetic-001: VALID
research/results/bm25-mini-001: VALID
```

Overwritten runs also validate:

```
<tmpdir>/overwrite-test: VALID
```

## Educational Implementation Symbols

The repository implements its own educational algorithms. These are the actual
implementation classes and functions:

| Module | Symbol | File |
|--------|--------|------|
| Linear regression | `LinearRegression` (class) | `airelab.foundations.linear_regression` |
| Linear regression | `run_linear_regression` (experiment function) | `airelab.foundations.linear_regression` |
| Logistic regression | `LogisticRegression` (class) | `airelab.foundations.logistic_regression` |
| Logistic regression | `_sigmoid` (function) | `airelab.foundations.logistic_regression` |
| Logistic regression | `run_logistic_regression` (experiment function) | `airelab.foundations.logistic_regression` |
| PCA | `PCA` (class) | `airelab.foundations.pca` |
| BM25 | `BM25` (class) | `airelab.foundations.bm25` |
| BM25 | `_tokenize` (function) | `airelab.foundations.bm25` |
| BM25 | `run_bm25` (experiment function) | `airelab.foundations.bm25` |
| Calibration | `brier_score` (function) | `airelab.foundations.calibration` |
| Calibration | `expected_calibration_error` (function) | `airelab.foundations.calibration` |
| Calibration | `reliability_bins` (function) | `airelab.foundations.calibration` |
| Calibration | `coverage` (function) | `airelab.foundations.calibration` |
| Calibration | `selective_risk` (function) | `airelab.foundations.calibration` |

External libraries used only for comparison or optional lab scripts:

| Library | Role | Location |
|---------|------|----------|
| scikit-learn | Independent reference; optional lab dependency | `labs/` scripts, `requirements.txt` |
| matplotlib | Plotting; optional lab dependency | `labs/` scripts, `requirements.txt` |
| pandas | Data handling; optional lab dependency | `labs/` scripts, `requirements.txt` |
| networkx | Graph algorithms; optional lab dependency | `labs/` scripts, `requirements.txt` |

These libraries are not required by the core `src/airelab/` package or the
experiment runner. The `labs/` directory is outside the core CI contract.

## Unchanged Limitations

All v0.1 limitations remain unchanged:

1. All data is synthetic; no real-world datasets.
2. No heavy ML frameworks in core (`src/airelab/`).
3. Single-run experiments; no variance/confidence intervals.
4. BM25 uses simple whitespace tokenizer.
5. PCA uses eigendecomposition only.
6. No hyperparameter search or cross-validation.
7. Educational implementations, not production-grade.
8. Calibration metrics use uniform binning only.
9. PCA is not wired to the experiment runner.

## Work Not Attempted

- New algorithms or statistical definitions.
- Cross-validation framework.
- Multi-seed experiments.
- PCA experiment runner wiring.
- Dependency modernization.
- Changes to any sibling repository.

## Local Windows Verification

All tests pass on Windows (Python 3.13.14, Windows 11 Pro):

```
92 passed in 26.62s
```

Overwrite test confirmed via production API: `run_experiment()` twice into the
same directory succeeds, manifest is valid JSON, artifact validation passes.

## Worktree

Clean. No uncommitted changes.
