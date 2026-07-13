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
