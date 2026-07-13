"""Tests for airelab.foundations.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from airelab.foundations.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
    selective_risk,
    coverage,
)


@pytest.mark.unit
class TestCalibration:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_overconfident_wrong(self) -> None:
        y_true = np.array([1, 0])
        y_prob = np.array([0.0, 1.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_hand_computed_brier(self) -> None:
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.8, 0.3, 0.6])
        # Brier = mean((y_true - y_prob)^2)
        expected = ((1 - 0.8) ** 2 + (0 - 0.3) ** 2 + (1 - 0.6) ** 2) / 3
        assert abs(brier_score(y_true, y_prob) - expected) < 1e-10

    def test_empty_accepted_set(self) -> None:
        result = reliability_bins(np.array([]), np.array([]), n_bins=5)
        assert len(result) == 0

    def test_non_finite_rejected(self) -> None:
        y_true = np.array([1, 0])
        y_prob = np.array([0.5, float("nan")])
        with pytest.raises(ValueError, match="finite"):
            brier_score(y_true, y_prob)

    def test_ece_perfect_calibration(self) -> None:
        """Perfect predictions should have ECE near 0."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=2)
        assert ece < 1e-10

    def test_reliability_bins_output(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
        bins = reliability_bins(y_true, y_prob, n_bins=3)
        assert len(bins) > 0
        for b in bins:
            assert "bin_start" in b
            assert "bin_end" in b
            assert "count" in b
            assert "positive_rate" in b
            assert "mean_predicted" in b

    def test_coverage(self) -> None:
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        cov = coverage(y_prob, threshold=0.5)
        assert cov == 0.5  # 2 of 4 above 0.5

    def test_selective_risk(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        y_pred = np.array([1, 0, 1, 0])
        sr = selective_risk(y_true, y_pred, y_prob, threshold=0.5)
        # Accepted: indices 0,2 (prob >= 0.5), predictions 1,1, true 1,1 → 0 errors
        assert sr == 0.0
