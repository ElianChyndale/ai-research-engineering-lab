"""Educational calibration metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _check_finite(*arrays: NDArray[np.float64]) -> None:
    for arr in arrays:
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input contains non-finite values")


def brier_score(y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> float:
    """Mean squared error between true labels and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    _check_finite(y_true, y_prob)
    return float(np.mean((y_true - y_prob) ** 2))


def expected_calibration_error(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) via uniform binning."""
    bins = reliability_bins(y_true, y_prob, n_bins)
    if not bins:
        return 0.0
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        weight = b["count"] / total
        ece += weight * abs(b["positive_rate"] - b["mean_predicted"])
    return float(ece)


def reliability_bins(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Compute reliability diagram bins.

    Returns list of dicts with keys:
      bin_start, bin_end, count, positive_rate, mean_predicted
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) == 0:
        return []

    _check_finite(y_true, y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    result: list[dict[str, float]] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(np.sum(mask))
        if count > 0:
            positive_rate = float(np.mean(y_true[mask]))
            mean_predicted = float(np.mean(y_prob[mask]))
        else:
            positive_rate = 0.0
            mean_predicted = (lo + hi) / 2.0
        result.append({
            "bin_start": float(lo),
            "bin_end": float(hi),
            "count": float(count),
            "positive_rate": positive_rate,
            "mean_predicted": mean_predicted,
        })
    return result


def coverage(y_prob: NDArray[np.float64], threshold: float = 0.5) -> float:
    """Fraction of predictions above the threshold."""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    _check_finite(y_prob)
    return float(np.mean(y_prob >= threshold))


def selective_risk(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    threshold: float = 0.5,
) -> float:
    """Error rate among accepted (high-confidence) predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    _check_finite(y_true, y_pred, y_prob)

    accepted = y_prob >= threshold
    n_accepted = int(np.sum(accepted))
    if n_accepted == 0:
        return float("nan")
    errors = np.sum(y_true[accepted] != y_pred[accepted])
    return float(errors / n_accepted)
