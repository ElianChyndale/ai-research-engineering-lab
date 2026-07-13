"""Statistical aggregation for multi-seed and repeated evaluation results.

Conventions:
- Standard deviation uses sample std (ddof=1) via statistics.stdev().
- Boolean values are NOT treated as numeric; they use all/any policy.
- Non-finite values (NaN, Inf) are rejected.
- Missing metrics follow a documented policy (aggregate over available values).
"""

from __future__ import annotations

import math
import statistics
from typing import Any


def aggregate_floats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of finite floats.

    Returns dict with keys: mean, std, median, min, max.
    Uses sample standard deviation (ddof=1).

    Policies:
    - Empty list: returns empty dict {}.
    - Single-element list: std=0.0, min=max=median=mean.
    - All values must be finite (no NaN, Inf, -Inf).
    """
    if not values:
        return {}
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"aggregate_floats requires finite values, got {v}")
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "std": 0.0, "median": v, "min": v, "max": v}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),  # sample std, ddof=1
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of metrics dicts.

    Policies:
    - Finite int/float: aggregated via aggregate_floats (mean, std, median, min, max).
    - bool: NOT aggregated as numeric. Reports {"all_true": bool, "any_true": bool}.
    - str/list/dict: first value is taken (not aggregated).
    - Non-finite (NaN, Inf): raises ValueError.
    - Missing keys: aggregated over available values only.
    - Empty list: returns empty dict.
    """
    if not metrics_list:
        return {}

    all_keys: set[str] = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    result: dict[str, Any] = {}
    for key in sorted(all_keys):
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue

        # Check types
        first = values[0]

        # bool must be checked before int (bool is subclass of int in Python)
        if isinstance(first, bool):
            if not all(isinstance(v, bool) for v in values):
                raise ValueError(f"Mixed bool/non-bool for key '{key}'")
            result[key] = {"all_true": all(values), "any_true": any(values)}
        elif all(isinstance(v, (int, float)) for v in values):
            float_vals = [float(v) for v in values]
            for v in float_vals:
                if not math.isfinite(v):
                    raise ValueError(
                        f"Non-finite value for key '{key}': {v}. "
                        "Use a structured non-evaluable representation instead."
                    )
            result[key] = aggregate_floats(float_vals)
        else:
            # Non-numeric (str, list, dict, None): take first value
            result[key] = first

    return result
