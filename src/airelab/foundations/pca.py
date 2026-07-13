"""Principal Component Analysis via eigendecomposition."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from airelab.core.config import ExperimentConfig
from airelab.experiments.registry import register


class PCA:
    """Educational PCA implementation.

    Attributes:
        components: Principal axes (n_components, n_features).
        explained_variance: Variance of each component.
        explained_variance_ratio: Fraction of total variance per component.
        mean: Per-feature mean used for centering.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components: NDArray[np.float64] | None = None
        self.explained_variance: NDArray[np.float64] | None = None
        self.explained_variance_ratio: NDArray[np.float64] | None = None
        self.mean: NDArray[np.float64] | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if self.n_components > d:
            raise ValueError(
                f"n_components ({self.n_components}) must be <= n_features ({d})"
            )

        # Center
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix eigendecomposition
        cov = (X_centered.T @ X_centered) / (n - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Fix sign convention: largest absolute element in each component is positive
        for i in range(eigenvectors.shape[1]):
            col = eigenvectors[:, i]
            max_idx = np.argmax(np.abs(col))
            if col[max_idx] < 0:
                eigenvectors[:, i] *= -1

        # Take top n_components
        self.components = eigenvectors[:, : self.n_components].T
        self.explained_variance = eigenvalues[: self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = (
            self.explained_variance / total_var if total_var > 0 else np.zeros(self.n_components)
        )
        self._fitted = True

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("PCA is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("PCA is not fitted. Call fit() first.")
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components + self.mean


@register("pca")
def run_pca(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Run a synthetic PCA experiment with known correlation structure."""
    params = config.parameters
    n_samples = int(params.get("n_samples", 100))
    n_features = int(params.get("n_features", 5))
    n_components = int(params.get("n_components", 2))

    # Generate synthetic data with known correlation:
    # feature 0 drives most variance, features 1..n are noisy copies
    np.random.seed(config.seed)
    base = np.random.randn(n_samples)
    noise = np.random.randn(n_samples, n_features) * 0.3
    X = np.column_stack([base * 3.0] + [base * (2.0 - i * 0.3) for i in range(n_features - 1)]) + noise

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Transform and reconstruct
    Z = pca.transform(X)
    X_recon = pca.inverse_transform(Z)
    reconstruction_mse = float(np.mean((X - X_recon) ** 2))

    # Write component loadings CSV
    csv_path = run_dir / "components.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["feature"] + [f"pc{i}" for i in range(n_components)]
        writer.writerow(header)
        for j in range(n_features):
            row = [f"x{j}"] + [float(pca.components[i, j]) for i in range(n_components)]
            writer.writerow(row)

    metrics = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio],
        "total_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio)),
        "reconstruction_mse": reconstruction_mse,
        "limitations": [
            "Synthetic data only",
            "Eigendecomposition only (no randomized SVD)",
            "Single seed — no variance estimate",
        ],
    }
    return metrics
