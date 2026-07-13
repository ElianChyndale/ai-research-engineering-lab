"""Principal Component Analysis via eigendecomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
