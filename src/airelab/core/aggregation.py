"""Statistical aggregation for multi-seed and repeated evaluation results."""

from __future__ import annotations

import math
import statistics
from typing import Any


def aggregate_floats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of floats.

    Returns dict with keys: mean, std, median, min, max.
    Single-element list: std=0.0, min=max=median=mean.
    Empty list: returns empty dict.
    """
    if not values:
        return {}
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "std": 0.0, "median": v, "min": v, "max": v}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of metrics dicts.

    For each numeric key present in all dicts, compute aggregate_floats.
    For non-numeric or inconsistent keys, take the first value.
    Empty list returns empty dict.
    """
    if not metrics_list:
        return {}

    # Collect all keys
    all_keys: set[str] = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    result: dict[str, Any] = {}
    for key in sorted(all_keys):
        values = [m[key] for m in metrics_list if key in m]
        # Check if all values are numeric
        if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
            result[key] = aggregate_floats([float(v) for v in values])
        else:
            # Non-numeric: take first value
            result[key] = values[0] if values else None

    return result
