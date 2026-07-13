"""Repeated train/test evaluation with different data splits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from airelab.core.aggregation import aggregate_floats, aggregate_metrics
from airelab.core.config import ExperimentConfig, ExperimentType
from airelab.core.seeds import set_seed


@dataclass(frozen=True)
class RepeatedEvalConfig:
    """Configuration for repeated train/test evaluation."""

    base_config: ExperimentConfig
    n_splits: int = 5
    test_size: float = 0.3
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    output_dir: str = ""

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.n_splits}")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        for seed in self.seeds:
            if seed < 0:
                raise ValueError(f"Seed must be non-negative, got {seed}")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if ".." in self.output_dir:
            raise ValueError(f"Path traversal detected in output_dir: {self.output_dir!r}")


def _generate_data(
    experiment_type: ExperimentType,
    params: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for the given experiment type."""
    set_seed(seed)
    n_samples = int(params.get("n_samples", 100))
    n_features = int(params.get("n_features", 3))

    if experiment_type == ExperimentType.LINEAR_REGRESSION:
        X = np.random.randn(n_samples, n_features)
        true_w = np.random.randn(n_features)
        noise_std = params.get("noise_std", 0.5)
        y = X @ true_w + 1.0 + np.random.randn(n_samples) * noise_std
    elif experiment_type == ExperimentType.LOGISTIC_REGRESSION:
        X = np.random.randn(n_samples, n_features)
        true_w = np.random.randn(n_features)
        logits = X @ true_w
        y = (logits > 0).astype(np.float64)
    else:
        raise ValueError(f"Repeated eval not supported for {experiment_type.value}")

    return X, y


def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets deterministically."""
    n = len(X)
    n_test = max(1, int(n * test_size))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _evaluate_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train linear regression and evaluate on test set."""
    from airelab.foundations.linear_regression import LinearRegression

    lr = params.get("learning_rate", 0.01)
    max_iter = int(params.get("max_iter", 5000))

    model = LinearRegression(solver="gradient_descent", learning_rate=lr, max_iter=max_iter)
    model.fit(X_train, y_train)

    return {
        "train_mse": model.mse(X_train, y_train),
        "test_mse": model.mse(X_test, y_test),
        "train_r2": model.r2(X_train, y_train),
        "test_r2": model.r2(X_test, y_test),
    }


def _evaluate_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train logistic regression and evaluate on test set."""
    from airelab.foundations.logistic_regression import LogisticRegression

    lr = params.get("learning_rate", 0.5)
    max_iter = int(params.get("max_iter", 1000))

    model = LogisticRegression(learning_rate=lr, max_iter=max_iter)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)

    return {
        "train_accuracy": float(np.mean(train_pred == y_train.astype(int))),
        "test_accuracy": float(np.mean(test_pred == y_test.astype(int))),
        "converged": model.converged,
    }


_EVALUATORS = {
    ExperimentType.LINEAR_REGRESSION: _evaluate_linear_regression,
    ExperimentType.LOGISTIC_REGRESSION: _evaluate_logistic_regression,
}


def run_repeated_eval(config: RepeatedEvalConfig) -> dict[str, Any]:
    """Run repeated train/test evaluation with different splits.

    For each seed, generates data, splits into train/test, trains, evaluates.
    Writes summary.json with aggregated metrics across splits.
    """
    experiment_type = config.base_config.experiment_type
    if experiment_type not in _EVALUATORS:
        raise ValueError(f"Repeated eval not supported for {experiment_type.value}")

    evaluator = _EVALUATORS[experiment_type]
    params = config.base_config.parameters

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split_metrics: list[dict[str, Any]] = []

    for i, seed in enumerate(config.seeds[: config.n_splits]):
        # Generate data with this seed
        X, y = _generate_data(experiment_type, params, seed)
        X_train, X_test, y_train, y_test = _train_test_split(X, y, config.test_size, seed)

        metrics = evaluator(X_train, y_train, X_test, y_test, params)
        metrics["split_seed"] = seed
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        per_split_metrics.append(metrics)

    aggregated = aggregate_metrics(per_split_metrics)

    summary = {
        "experiment_id": config.base_config.experiment_id,
        "experiment_type": experiment_type.value,
        "n_splits": min(config.n_splits, len(config.seeds)),
        "test_size": config.test_size,
        "seeds": list(config.seeds[: config.n_splits]),
        "per_split_metrics": per_split_metrics,
        "aggregated_metrics": aggregated,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    return summary
