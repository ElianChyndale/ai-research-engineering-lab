"""Tests for airelab.foundations.linear_regression."""

from __future__ import annotations

import numpy as np
import pytest

from airelab.foundations.linear_regression import LinearRegression


@pytest.mark.unit
class TestLinearRegression:
    def test_known_ols_example(self) -> None:
        """y = 2x + 1, exact fit."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X)
        np.testing.assert_allclose(pred, y, atol=1e-10)
        assert model.coefficients is not None
        assert abs(model.coefficients[0] - 2.0) < 1e-10
        assert abs(model.intercept - 1.0) < 1e-10

    def test_gradient_descent_agrees_with_ols(self) -> None:
        """Gradient descent should converge close to OLS solution."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_w = np.array([1.0, -2.0, 0.5])
        y = X @ true_w + 3.0 + np.random.randn(100) * 0.1

        ols = LinearRegression()
        ols.fit(X, y)

        gd = LinearRegression(solver="gradient_descent", learning_rate=0.01, max_iter=5000)
        gd.fit(X, y)

        np.testing.assert_allclose(gd.coefficients, ols.coefficients, atol=0.1)
        assert abs(gd.intercept - ols.intercept) < 0.5

    def test_singular_matrix_handled(self) -> None:
        """Duplicate features should not crash."""
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y = np.array([1.0, 2.0, 3.0])
        model = LinearRegression()
        # Should either fit or raise a clear error, not crash silently
        try:
            model.fit(X, y)
        except np.linalg.LinAlgError:
            pass  # acceptable

    def test_non_finite_input_rejected(self) -> None:
        X = np.array([[1.0], [float("nan")]])
        y = np.array([1.0, 2.0])
        model = LinearRegression()
        with pytest.raises(ValueError, match="finite"):
            model.fit(X, y)

    def test_non_finite_y_rejected(self) -> None:
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, float("inf")])
        model = LinearRegression()
        with pytest.raises(ValueError, match="finite"):
            model.fit(X, y)

    def test_deterministic_result(self) -> None:
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m1 = LinearRegression()
        m1.fit(X, y)
        m2 = LinearRegression()
        m2.fit(X, y)
        np.testing.assert_array_equal(m1.coefficients, m2.coefficients)
        assert m1.intercept == m2.intercept

    def test_mse_computation(self) -> None:
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([3.0, 5.0, 7.0])
        model = LinearRegression()
        model.fit(X, y)
        mse = model.mse(X, y)
        assert mse < 1e-20

    def test_r2_perfect_fit(self) -> None:
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([3.0, 5.0, 7.0])
        model = LinearRegression()
        model.fit(X, y)
        assert model.r2(X, y) > 0.9999

    def test_predict_before_fit_raises(self) -> None:
        model = LinearRegression()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.array([[1.0]]))
