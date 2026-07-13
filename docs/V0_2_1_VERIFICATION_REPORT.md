# AI Research Engineering Lab v0.2.1 — Verification Report

## Parent Commit

`886dd2e` — feat: add multi-seed, repeated evaluation, and cross-validation to v0.2

## Maintenance Commit

`<hash>` — fix: harden evaluation summaries and comparisons

## Independent Audit Result

V0.2 AUDIT: PASS_WITH_LIMITATIONS

Verified 116 tests pass, all v0.2 features functional, committed artifacts clean.

## Changes in v0.2.1

### 1. Strict Summary Artifact Validation

**Problem:** Multi-seed, repeated-evaluation, and cross-validation summaries were
written without validating that they contain no NaN or Infinity. Only run-level
`metrics.json` was validated.

**Fix:**
- Added `validate_summary()` and `validate_summary_file()` to `validation.py`.
- All summary-writing modules (`multi_seed.py`, `repeated_eval.py`,
  `cross_validation.py`) now validate before writing.
- Serialization uses `json.dumps(..., allow_nan=False)`.
- `validate_summary_file()` uses `parse_constant` to reject NaN/Infinity literals.
- Recursive validation identifies the exact path to any non-finite value.

**Tests:** 8 new summary validation tests (nested NaN, nested Inf, deeply nested,
strict JSON, file-level NaN literal, missing file).

### 2. Safe Aggregation Policy

**Problem:** `aggregate_metrics()` treated booleans as numeric (True=1, False=0)
and did not document the std convention. Non-finite values could silently propagate.

**Fix:**
- `aggregate_floats()` now raises `ValueError` on non-finite inputs.
- `aggregate_metrics()` uses explicit policies:
  - `bool`: `{"all_true": bool, "any_true": bool}` (not numeric)
  - `int/float`: aggregated via `aggregate_floats` (must be finite)
  - `str/list/dict`: first-value policy (not aggregated)
- Module docstring documents: "Standard deviation uses sample std (ddof=1)."

**Tests:** 10 new aggregation tests (ddof=1 verification, NaN rejection, Inf
rejection, bool policy, list policy).

### 3. Explicit Family Comparability

**Problem:** `compare_families()` did not check or flag whether two experiment
families were comparable. Different experiment types could be compared as if
they were equivalent.

**Fix:**
- Added `_check_comparability()` that returns `(comparable, reasons)`.
- Checks: experiment type, dataset identity, protocol (n_folds, n_splits,
  test_size, shuffle), fixture status.
- When `comparable=False`:
  - `mean_diff` is NOT computed (no superiority implication).
  - `conclusion` field explicitly states "No performance conclusion permitted."
  - Reasons are listed.
- When `comparable=True`: `mean_diff` is computed descriptively.

**Tests:** 6 new comparability tests (same family, different types, different
datasets, different protocols, different fixture status, no-winner implication).

### 4. Targeted Test Gaps

**Problem:** Several high-value behaviors had no test coverage.

**Fix:** 11 new integration tests covering:
- PCA deterministic semantic artifacts (metrics + components CSV).
- Repeated train/test indices do not overlap.
- Each CV observation appears in exactly one validation fold.
- Fresh model instance created for each CV fold (different weights).
- Valid multi-seed summary passes validation.
- Valid CV summary passes validation.
- Summary is strict-JSON-parseable (no NaN literals).
- `selective_risk` NaN cannot enter summary (verified isolation).
- Comparable family comparison works.
- Incompatible family comparison works.

### 5. Repeated-Evaluation Semantics

**Problem:** The documentation did not clarify that repeated evaluation changes
both synthetic data generation AND train/test split per seed.

**Fix:**
- Added module docstring to `repeated_eval.py`: "Each seed generates new
  synthetic data AND a new train/test split. The observed variance combines
  data-generation variance, split variance, and model-training variance."
- Updated `run_repeated_eval()` docstring.
- Updated README limitations.

**Not redesigned:** The implementation is preserved as-is. Cross-validation
remains distinct because it generates one dataset and changes folds.

### 6. PCA Seed Consistency

**Problem:** `run_pca()` used `np.random.seed(config.seed)` directly, while all
other experiments used `set_seed(config.seed)`.

**Fix:** Replaced with `set_seed(config.seed)`. Verified that semantic outputs
(metrics, components) are identical before and after.

### 7. SVG and Worktree Hygiene

**Problem:** Lab scripts (`01_ml_foundations`, `02_deep_learning`) generate SVG
files in `research/results/`, which is tracked by git. Each test run regenerated
these SVGs with different internal matplotlib IDs, causing dirty worktree state.

**Fix:**
- Added `research/results/*.svg` to `.gitignore`.
- Removed SVG files from git tracking (`git rm --cached`).
- SVG files are still generated on disk by lab scripts but are no longer tracked.
- After full test suite: `git status --short` shows no SVG modifications.

### 8. Documentation

- Updated README limitations with ddof=1 convention, repeated-eval semantics,
  and CV interpretation.
- Created this verification report.

## Test Results

```
151 passed in 44.18s
```

Breakdown:
- 116 inherited from v0.2 (unchanged)
- 10 new aggregation tests
- 8 new summary validation tests
- 6 new comparison tests
- 11 new integration tests

## Experiment Reruns

All committed run artifacts validated:
- 19 run directories (including seed subdirectories)
- All pass `validate_artifacts.py`
- No NaN or Infinity in any committed fixture

## Remaining Limitations

1. **No graceful failure in multi-seed.** If one seed fails, the entire run
   fails. No partial results or failure count.

2. **Repeated eval conflates variance sources.** Data-generation and split
   variance are combined. This is documented but not redesigned.

3. **Summary.json validation is in-memory.** The `validate_summary()` function
   validates a dict, not a file. `validate_summary_file()` handles files. Both
   are used by the writing modules before serialization.

4. **No stratification in CV.** Cross-validation does not stratify by class
   label. With balanced synthetic data this is acceptable.

5. **`converged` boolean aggregation changes format.** Previously aggregated as
   numeric (mean=0.667). Now aggregated as `{"all_true": False, "any_true": True}`.
   This is a breaking change for downstream consumers of the aggregated metric.

## Files Changed

| File | Change |
|------|--------|
| `src/airelab/core/aggregation.py` | Safe aggregation with bool/NaN/Inf policies |
| `src/airelab/core/validation.py` | Added `validate_summary()`, `validate_summary_file()` |
| `src/airelab/experiments/comparison.py` | Explicit comparability with reasons |
| `src/airelab/experiments/multi_seed.py` | Summary validation, dataset_id/fixture in output |
| `src/airelab/experiments/repeated_eval.py` | Summary validation, variance documentation |
| `src/airelab/experiments/cross_validation.py` | Summary validation, dataset_id/fixture in output |
| `src/airelab/foundations/pca.py` | Consistent `set_seed()` usage |
| `tests/unit/test_aggregation.py` | 10 new tests |
| `tests/unit/test_validation.py` | 8 new tests |
| `tests/unit/test_comparison.py` | New file, 6 tests |
| `tests/integration/test_evaluation_hardening.py` | New file, 11 tests |
| `README.md` | Updated limitations |
| `.gitignore` | Added `research/results/*.svg` |
