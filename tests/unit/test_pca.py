"""Tests for airelab.foundations.pca."""

from __future__ import annotations

import numpy as np
import pytest

from airelab.foundations.pca import PCA


@pytest.mark.unit
class TestPCA:
    def test_hand_computed_2d(self) -> None:
        """Points along x-axis: first PC should be x."""
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        pca = PCA(n_components=1)
        pca.fit(X)
        # Explained variance ratio should be ~1.0
        assert pca.explained_variance_ratio[0] > 0.99

    def test_reconstruction_improves_with_more_components(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 5)
        # Add correlation
        X[:, 1] = X[:, 0] * 2 + np.random.randn(50) * 0.1
        X[:, 2] = X[:, 0] * 3 + np.random.randn(50) * 0.1

        pca1 = PCA(n_components=1)
        pca1.fit(X)
        recon1 = pca1.inverse_transform(pca1.transform(X))
        err1 = np.mean((X - recon1) ** 2)

        pca3 = PCA(n_components=3)
        pca3.fit(X)
        recon3 = pca3.inverse_transform(pca3.transform(X))
        err3 = np.mean((X - recon3) ** 2)

        assert err3 < err1

    def test_explained_variance_ordered(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 4)
        pca = PCA(n_components=4)
        pca.fit(X)
        ratios = pca.explained_variance_ratio
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1]

    def test_invalid_component_count_fails(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pca = PCA(n_components=5)
        with pytest.raises(ValueError, match="n_components"):
            pca.fit(X)

    def test_repeated_run_deterministic(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        pca1 = PCA(n_components=2)
        pca1.fit(X)
        pca2 = PCA(n_components=2)
        pca2.fit(X)
        np.testing.assert_allclose(
            np.abs(pca1.components), np.abs(pca2.components), atol=1e-10
        )

    def test_transform_shape(self) -> None:
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        pca = PCA(n_components=2)
        pca.fit(X)
        Z = pca.transform(X)
        assert Z.shape == (4, 2)
