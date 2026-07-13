"""K-fold cross-validation experiment execution.

Generates data ONCE and partitions into k folds. Each fold serves as
validation once while the remaining k-1 folds form the training set.
Model state is recreated for every fold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from airelab.core.aggregation import aggregate_metrics
from airelab.core.config import ExperimentConfig, ExperimentType
from airelab.core.seeds import set_seed
from airelab.core.validation import validate_summary
from airelab.experiments.repeated_eval import (
    _generate_data,
    _evaluate_linear_regression,
    _evaluate_logistic_regression,
)


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for k-fold cross-validation."""

    base_config: ExperimentConfig
    n_folds: int = 5
    shuffle: bool = True
    seed: int = 42
    output_dir: str = ""

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if ".." in self.output_dir:
            raise ValueError(f"Path traversal detected in output_dir: {self.output_dir!r}")


_EVALUATORS = {
    ExperimentType.LINEAR_REGRESSION: _evaluate_linear_regression,
    ExperimentType.LOGISTIC_REGRESSION: _evaluate_logistic_regression,
}


def _make_folds(
    n: int,
    n_folds: int,
    shuffle: bool,
    seed: int,
) -> list[np.ndarray]:
    """Split indices into k folds, optionally shuffled."""
    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def run_cross_validation(config: CrossValidationConfig) -> dict[str, Any]:
    """Run k-fold cross-validation.

    Generates data ONCE, splits into k folds, trains on k-1, evaluates on 1.
    Model state is recreated for every fold (no state leakage).
    Writes summary.json with per-fold and aggregated metrics.
    Validates summary before writing.
    """
    experiment_type = config.base_config.experiment_type
    if experiment_type not in _EVALUATORS:
        raise ValueError(f"Cross-validation not supported for {experiment_type.value}")

    evaluator = _EVALUATORS[experiment_type]
    params = config.base_config.parameters

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate data once
    X, y = _generate_data(experiment_type, params, config.seed)

    # Make folds
    folds = _make_folds(len(X), config.n_folds, config.shuffle, config.seed)

    per_fold_metrics: list[dict[str, Any]] = []

    for fold_idx in range(config.n_folds):
        # Test = fold_idx, Train = all other folds
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(config.n_folds) if i != fold_idx])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fresh model instance for each fold — no state leakage
        metrics = evaluator(X_train, y_train, X_test, y_test, params)
        metrics["fold"] = fold_idx
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        per_fold_metrics.append(metrics)

    aggregated = aggregate_metrics(per_fold_metrics)

    summary = {
        "experiment_id": config.base_config.experiment_id,
        "experiment_type": experiment_type.value,
        "dataset_id": config.base_config.dataset_id,
        "fixture": config.base_config.fixture,
        "n_folds": config.n_folds,
        "shuffle": config.shuffle,
        "seed": config.seed,
        "per_fold_metrics": per_fold_metrics,
        "aggregated_metrics": aggregated,
    }

    # Validate before writing — reject non-finite values
    result = validate_summary(summary)
    if not result.valid:
        raise ValueError(f"Summary validation failed:\n{result}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return summary
