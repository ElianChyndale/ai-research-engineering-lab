"""Binary logistic regression with L2 regularization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from airelab.core.config import ExperimentConfig
from airelab.experiments.registry import register


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """Educational binary logistic regression with L2 regularization.

    Attributes:
        coefficients: Weight vector (n_features,).
        intercept: Scalar bias term.
        converged: Whether the solver converged within max_iter.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        l2_lambda: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2_lambda = l2_lambda
        self.coefficients: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.converged: bool = False
        self._fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.size == 0 or y.size == 0:
            raise ValueError("Data must not be empty")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(
                f"Need at least 2 classes, got {len(classes)}: {classes.tolist()}"
            )

        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        self.converged = False
        for iteration in range(self.max_iter):
            z = X @ w + b
            prob = _sigmoid(z)
            error = prob - y

            grad_w = (1.0 / n) * (X.T @ error) + self.l2_lambda * w
            grad_b = (1.0 / n) * np.sum(error)

            w_new = w - self.learning_rate * grad_w
            b_new = b - self.learning_rate * grad_b

            if np.max(np.abs(w_new - w)) < self.tol and abs(b_new - b) < self.tol:
                self.converged = True
                w, b = w_new, b_new
                break
            w, b = w_new, b_new

        self.coefficients = w
        self.intercept = float(b)
        self._fitted = True

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        z = X @ self.coefficients + self.intercept
        return _sigmoid(z)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        return (self.predict_proba(X) >= 0.5).astype(int)


@register("logistic_regression")
def run_logistic_regression(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Run a synthetic logistic regression experiment."""
    params = config.parameters
    n_samples = int(params.get("n_samples", 100))
    n_features = int(params.get("n_features", 2))

    # Generate synthetic binary classification data
    np.random.seed(config.seed)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    logits = X @ true_w
    y = (logits > 0).astype(np.float64)

    # Fit
    lr = params.get("learning_rate", 0.5)
    max_iter = int(params.get("max_iter", 1000))
    model = LogisticRegression(learning_rate=lr, max_iter=max_iter)
    model.fit(X, y)

    proba = model.predict_proba(X)
    pred = model.predict(X)

    # Accuracy
    accuracy = float(np.mean(pred == y.astype(int)))

    # Write predictions CSV
    csv_path = run_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(n_features)] + ["y_true", "predicted", "probability"]
        writer.writerow(header)
        for i in range(n_samples):
            writer.writerow([float(X[i, j]) for j in range(n_features)]
                            + [int(y[i]), int(pred[i]), float(proba[i])])

    metrics = {
        "accuracy": accuracy,
        "converged": model.converged,
        "n_samples": n_samples,
        "n_features": n_features,
        "limitations": [
            "Synthetic data only",
            "No train/test split",
            "Single seed — no variance estimate",
        ],
    }
    return metrics
