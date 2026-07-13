"""Compare two experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    """Compare two experiment run directories.

    Returns a dict with config_diff, metric_diff, artifact_diff, env_diff,
    and a meaningfulness flag.
    """
    manifest_a = _load_json(run_a / "manifest.json")
    manifest_b = _load_json(run_b / "manifest.json")

    config_a = _load_json(run_a / "config.json")
    config_b = _load_json(run_b / "config.json")

    metrics_a = _load_json(run_a / "metrics.json")
    metrics_b = _load_json(run_b / "metrics.json")

    env_a = _load_json(run_a / "environment.json")
    env_b = _load_json(run_b / "environment.json")

    config_diff = _dict_diff(config_a, config_b)
    metric_diff = _dict_diff(metrics_a, metrics_b)
    env_diff = _dict_diff(env_a, env_b)

    # Artifact hash comparison
    artifacts_a = {a["path"]: a["sha256"] for a in manifest_a.get("artifacts", [])}
    artifacts_b = {a["path"]: a["sha256"] for a in manifest_b.get("artifacts", [])}
    artifact_diff = _dict_diff(artifacts_a, artifacts_b)

    # Is comparison meaningful?
    same_type = manifest_a.get("experiment_type") == manifest_b.get("experiment_type")
    same_seed = manifest_a.get("seed") == manifest_b.get("seed")
    meaningful = same_type

    return {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "same_experiment_type": same_type,
        "same_seed": same_seed,
        "config_differences": config_diff,
        "metric_differences": metric_diff,
        "artifact_differences": artifact_diff,
        "environment_differences": env_diff,
        "comparison_meaningful": meaningful,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return keys that differ between two dicts."""
    all_keys = set(a) | set(b)
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        val_a = a.get(key)
        val_b = b.get(key)
        if val_a != val_b:
            diff[key] = {"a": val_a, "b": val_b}
    return diff


def compare_families(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    """Compare two multi-seed experiment family directories.

    Each directory must contain summary.json with aggregated_metrics.
    Returns comparison of aggregated metrics across families.
    """
    summary_a = _load_json(dir_a / "summary.json")
    summary_b = _load_json(dir_b / "summary.json")

    agg_a = summary_a.get("aggregated_metrics", {})
    agg_b = summary_b.get("aggregated_metrics", {})

    # Compare numeric aggregated metrics
    all_keys = set(agg_a.keys()) | set(agg_b.keys())
    metric_comparisons: dict[str, dict[str, Any]] = {}

    for key in sorted(all_keys):
        val_a = agg_a.get(key)
        val_b = agg_b.get(key)

        # Both must be aggregated dicts with mean/std
        if isinstance(val_a, dict) and isinstance(val_b, dict) and "mean" in val_a and "mean" in val_b:
            mean_a = val_a["mean"]
            mean_b = val_b["mean"]
            range_a = (val_a.get("min", mean_a), val_a.get("max", mean_a))
            range_b = (val_b.get("min", mean_b), val_b.get("max", mean_b))
            # Ranges overlap if max of one >= min of the other
            ranges_overlap = range_a[1] >= range_b[0] and range_b[1] >= range_a[0]
            metric_comparisons[key] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_diff": mean_a - mean_b,
                "std_a": val_a.get("std", 0.0),
                "std_b": val_b.get("std", 0.0),
                "range_a": list(range_a),
                "range_b": list(range_b),
                "ranges_overlap": ranges_overlap,
            }
        else:
            metric_comparisons[key] = {
                "value_a": val_a,
                "value_b": val_b,
                "equal": val_a == val_b,
            }

    return {
        "family_a": str(dir_a),
        "family_b": str(dir_b),
        "experiment_type_a": summary_a.get("experiment_type"),
        "experiment_type_b": summary_b.get("experiment_type"),
        "seeds_a": summary_a.get("seeds", []),
        "seeds_b": summary_b.get("seeds", []),
        "metric_comparisons": metric_comparisons,
    }
