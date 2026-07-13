"""Ordinary least squares and gradient descent linear regression."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from airelab.core.config import ExperimentConfig
from airelab.experiments.registry import register


class LinearRegression:
    """Educational linear regression with OLS and gradient descent solvers.

    Attributes:
        coefficients: Weight vector (n_features,).
        intercept: Scalar bias term.
    """

    def __init__(
        self,
        solver: str = "ols",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self._fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")

        if self.solver == "ols":
            self._fit_ols(X, y)
        elif self.solver == "gradient_descent":
            self._fit_gd(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver!r}")
        self._fitted = True

    def _fit_ols(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        n, d = X.shape
        X_aug = np.column_stack([X, np.ones(n)])
        try:
            w = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            raise
        self.coefficients = w[:d]
        self.intercept = float(w[d])

    def _fit_gd(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.max_iter):
            pred = X @ w + b
            error = pred - y
            grad_w = (2.0 / n) * (X.T @ error)
            grad_b = (2.0 / n) * np.sum(error)
            w_new = w - self.learning_rate * grad_w
            b_new = b - self.learning_rate * grad_b
            if np.max(np.abs(w_new - w)) < self.tol and abs(b_new - b) < self.tol:
                w, b = w_new, b_new
                break
            w, b = w_new, b_new
        self.coefficients = w
        self.intercept = float(b)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coefficients + self.intercept

    def mse(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        pred = self.predict(X)
        return float(np.mean((pred - y) ** 2))

    def r2(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


@register("linear_regression")
def run_linear_regression(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Run a synthetic linear regression experiment."""
    params = config.parameters
    n_samples = int(params.get("n_samples", 100))
    n_features = int(params.get("n_features", 3))
    noise_std = params.get("noise_std", 0.5)

    # Generate synthetic data with known coefficients
    np.random.seed(config.seed)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    true_b = 1.0
    y = X @ true_w + true_b + np.random.randn(n_samples) * noise_std

    # Fit OLS
    ols = LinearRegression(solver="ols")
    ols.fit(X, y)

    # Fit GD
    lr = params.get("learning_rate", 0.01)
    max_iter = int(params.get("max_iter", 5000))
    gd = LinearRegression(solver="gradient_descent", learning_rate=lr, max_iter=max_iter)
    gd.fit(X, y)

    # Predictions
    ols_pred = ols.predict(X)
    gd_pred = gd.predict(X)

    # Write predictions CSV
    csv_path = run_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(n_features)] + ["y_true", "ols_pred", "gd_pred"]
        writer.writerow(header)
        for i in range(n_samples):
            row = list(X[i]) + [float(y[i]), float(ols_pred[i]), float(gd_pred[i])]
            writer.writerow(row)

    # Metrics
    metrics = {
        "ols_mse": ols.mse(X, y),
        "ols_r2": ols.r2(X, y),
        "gd_mse": gd.mse(X, y),
        "gd_r2": gd.r2(X, y),
        "n_samples": n_samples,
        "n_features": n_features,
        "limitations": [
            "Synthetic data only",
            "No train/test split",
            "Single seed — no variance estimate",
        ],
    }
    return metrics
