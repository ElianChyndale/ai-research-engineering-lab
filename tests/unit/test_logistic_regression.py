"""Tests for airelab.foundations.logistic_regression."""

from __future__ import annotations

import numpy as np
import pytest

from airelab.foundations.logistic_regression import LogisticRegression


@pytest.mark.unit
class TestLogisticRegression:
    def test_hand_computed_probabilities(self) -> None:
        """Perfectly separated data should give probabilities near 0/1."""
        X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression(learning_rate=1.0, max_iter=1000)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert all(0.0 <= p <= 1.0 for p in proba)
        assert all(p < 0.1 for p in proba[:3])
        assert all(p > 0.9 for p in proba[3:])

    def test_label_change_alters_coefficients(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y1 = np.array([0, 0, 1, 1])
        y2 = np.array([1, 1, 0, 0])
        m1 = LogisticRegression(learning_rate=0.1, max_iter=500)
        m1.fit(X, y1)
        m2 = LogisticRegression(learning_rate=0.1, max_iter=500)
        m2.fit(X, y2)
        # Coefficients should have opposite signs
        assert m1.coefficients[0] * m2.coefficients[0] < 0

    def test_empty_data_fails(self) -> None:
        model = LogisticRegression()
        with pytest.raises(ValueError, match="empty"):
            model.fit(np.array([]).reshape(0, 1), np.array([]))

    def test_one_class_data_fails(self) -> None:
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0])
        model = LogisticRegression()
        with pytest.raises(ValueError, match="class"):
            model.fit(X, y)

    def test_predict_proba_range(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.5, max_iter=1000)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert np.all(np.isfinite(proba))

    def test_predict_output(self) -> None:
        X = np.array([[0.0], [0.1], [10.0], [10.1]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(learning_rate=1.0, max_iter=1000)
        model.fit(X, y)
        pred = model.predict(X)
        assert set(pred).issubset({0, 1})

    def test_convergence_status(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(learning_rate=0.01, max_iter=5)
        model.fit(X, y)
        assert hasattr(model, "converged")

    def test_deterministic(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        m1 = LogisticRegression(learning_rate=0.1, max_iter=500)
        m1.fit(X, y)
        m2 = LogisticRegression(learning_rate=0.1, max_iter=500)
        m2.fit(X, y)
        np.testing.assert_array_equal(m1.coefficients, m2.coefficients)
