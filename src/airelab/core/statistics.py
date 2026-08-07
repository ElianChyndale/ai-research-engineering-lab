"""Reusable statistics for multi-seed research experiments.

Covers the needs that recurred in the falsification programme:
  - bootstrap confidence intervals,
  - paired differences with aligned seeds,
  - practical effect size and equivalence margins,
  - a small effect-size estimate (Cohen's d).

Do NOT overbuild hypothesis testing — effect size and regime replication
matter more in pre-registered cheap-kill experiments.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable

import numpy as np


def bootstrap_ci(
    values: list[float],
    *,
    stat: Callable[[list[float]], float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI (default: mean) for a list of finite values.

    Returns (lo, hi). Empty input -> (nan, nan). Non-finite values are dropped.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    fn = stat or (lambda xs: float(np.mean(xs)))
    rng = np.random.default_rng(seed)
    boots = [fn(rng.choice(x, size=x.size, replace=True).tolist()) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_differences(a: list[float], b: list[float], *, seeds_a: list[int] | None = None,
                       seeds_b: list[int] | None = None) -> dict[str, object]:
    """Paired differences aligned by seed (method A - method B).

    If seeds are provided, they must be the same length and matched by position;
    otherwise pairs are matched by index. Returns mean/median/CI of the
    differences and the per-seed pairs.
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must match: {len(a)} vs {len(b)}")
    if seeds_a is not None and seeds_b is not None:
        if seeds_a != seeds_b:
            raise ValueError("seeds must be aligned and identical for a paired test")
    diffs = [float(x - y) for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": len(diffs),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "ci_lo": lo,
        "ci_hi": hi,
        "pairs": diffs,
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled Cohen's d effect size between two groups."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    sp = math.sqrt(
        ((x.size - 1) * float(np.var(x, ddof=1)) + (y.size - 1) * float(np.var(y, ddof=1)))
        / (x.size + y.size - 2)
    )
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def beats_by_margin(a: list[float], b: list[float], *, margin: float = 0.25) -> bool:
    """Does A beat B by a practical margin on the MEAN (lower-is-better metric)?

    A beats B by margin if mean(A) < (1 - margin) * mean(B), i.e. A is at least
    `margin`-fraction better. This encodes the pre-registered effect-size rule
    used in the decisive experiment (>=25% lower N_eps).
    """
    ma = float(np.nanmean(a))
    mb = float(np.nanmean(b))
    if mb == 0:
        return False
    return ma < (1.0 - margin) * mb


def equivalence_within(a: list[float], b: list[float], *, tol: float = 0.05) -> bool:
    """Practical equivalence: |mean(A) - mean(B)| / max(|mean(A)|,|mean(B)|,1) <= tol."""
    ma = float(np.nanmean(a))
    mb = float(np.nanmean(b))
    denom = max(abs(ma), abs(mb), 1.0)
    return abs(ma - mb) / denom <= tol
