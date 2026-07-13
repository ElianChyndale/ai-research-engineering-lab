# V0.1 Verification Report

## Commits

| Hash | Message |
|------|---------|
| `713d9d3` | chore: establish research engineering laboratory |
| `f674c5a` | docs: add research learning workflow |

## Files

- 117 files in initial commit
- 8 doc templates in second commit
- Total tracked files: ~95 (excluding generated artifacts not in commit)

## Tests

```
89 passed in 24.85s
```

### Unit Tests (80)

| Module | Tests |
|--------|-------|
| test_config | 10 |
| test_seeds | 5 |
| test_artifacts | 5 |
| test_manifest | 8 |
| test_validation | 10 |
| test_linear_regression | 9 |
| test_logistic_regression | 8 |
| test_pca | 6 |
| test_bm25 | 9 |
| test_calibration | 9 |
| test_placeholder | 1 |

### Integration Tests (7)

| Test | Description |
|------|-------------|
| test_linear_regression_experiment | Full run via subprocess |
| test_logistic_regression_experiment | Full run via subprocess |
| test_bm25_experiment | Full run via subprocess |
| test_deterministic_rerun | Same config+seed → identical metrics |
| test_validate_valid_run | VALID result on clean run |
| test_validate_tampered_run_fails | Hash mismatch detected |
| test_compare_runs | Comparison reports differences |

## Commands Run

```bash
python scripts/run_experiment.py --config configs/linear_regression.yaml
python scripts/run_experiment.py --config configs/logistic_regression.yaml
python scripts/run_experiment.py --config configs/bm25.yaml
python scripts/validate_artifacts.py --run-dir research/results/lr-synthetic-001
python scripts/validate_artifacts.py --run-dir research/results/logr-synthetic-001
python scripts/validate_artifacts.py --run-dir research/results/bm25-mini-001
python scripts/compare_runs.py <run-a> <run-b>
python -m pytest -q
```

## Generated Artifact Schemas

### config.json
```json
{
  "dataset_id": "...",
  "experiment_id": "...",
  "experiment_type": "...",
  "fixture": true,
  "notes": "...",
  "output_dir": "...",
  "parameters": {},
  "seed": 42
}
```

### metrics.json
Per-experiment keys. Linear regression: `ols_mse`, `ols_r2`, `gd_mse`, `gd_r2`. Logistic regression: `accuracy`, `converged`. BM25: `n_documents`, `n_queries`, `mean_score`, `max_score`.

### predictions.csv / rankings.csv
CSV with headers. Predictions: feature columns + `y_true` + model predictions. Rankings: `query`, `rank`, `doc_id`, `score`, `document`.

### environment.json
```json
{
  "dependencies": {"numpy": "2.1.2", "pyyaml": "..."},
  "platform": "...",
  "python_major_minor": "3.13",
  "python_version": "..."
}
```

### manifest.json
```json
{
  "schema_version": 1,
  "experiment_id": "...",
  "experiment_type": "...",
  "seed": 42,
  "git_commit": "...",
  "git_dirty": false,
  "command": "...",
  "configuration": {},
  "python_version": "...",
  "dependencies": {},
  "start_time": "...",
  "end_time": "...",
  "fixture": true,
  "artifacts": [{"path": "...", "sha256": "...", "size": 0}],
  "success": true
}
```

## Deterministic Rerun Result

All three experiments run twice in separate temp directories:

| Experiment | Run A metrics | Run B metrics | Result |
|-----------|---------------|---------------|--------|
| linear_regression | match | match | DETERMINISTIC |
| logistic_regression | match | match | DETERMINISTIC |
| bm25 | match | match | DETERMINISTIC |

## Known Limitations

1. All data is synthetic; no real-world datasets
2. No heavy ML frameworks (scikit-learn, PyTorch, TensorFlow)
3. Single-run experiments; no variance/confidence intervals
4. BM25 uses simple whitespace tokenizer (no stemming/stop-words)
5. PCA uses eigendecomposition only (no randomized SVD for large matrices)
6. No hyperparameter search or cross-validation
7. Educational implementations, not production-grade
8. Calibration metrics use uniform binning only

## Work Not Attempted

- PCA experiment config and runner registration (PCA module exists but is not wired to experiment runner)
- Statistical significance testing
- Cross-validation framework
- Hyperparameter search
- Real dataset loading
- Heavy ML framework integration

## Items Requiring Stronger-Model Review

1. BM25 IDF formula — verify against original Robertson et al. paper
2. PCA sign convention — verify the "largest absolute element positive" convention is standard
3. Logistic regression gradient — verify the L2 regularization gradient is correct
4. Calibration ECE computation — verify binning strategy matches standard definition

## Dirty Status

Clean (all changes committed).

## Recommended Next Task

- Wire PCA to experiment runner with a config
- Add cross-validation support to experiment runner
- Add variance reporting (multiple seeds) to experiments
