"""Artifact validation for experiment runs and summaries."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airelab.core.artifacts import hash_file


REQUIRED_ARTIFACTS = ("config.json", "metrics.json", "manifest.json")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]

    def __str__(self) -> str:
        if self.valid:
            return "VALID"
        return "INVALID:\n" + "\n".join(f"  - {e}" for e in self.errors)


def validate_run(run_dir: Path) -> ValidationResult:
    """Validate an experiment run directory.

    Checks:
    - Required artifacts exist
    - manifest.json is valid JSON
    - Artifact hashes in manifest match actual files
    - No NaN/Infinity in metrics
    - Fixture runs cannot be labelled reviewed
    """
    errors: list[str] = []

    if not run_dir.is_dir():
        return ValidationResult(valid=False, errors=[f"Not a directory: {run_dir}"])

    # Check required artifacts
    for name in REQUIRED_ARTIFACTS:
        p = run_dir / name
        if not p.exists():
            errors.append(f"Missing required artifact: {name}")

    if errors:
        return ValidationResult(valid=False, errors=errors)

    # Load and validate manifest
    manifest_path = run_dir / "manifest.json"
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return ValidationResult(valid=False, errors=[f"Invalid manifest.json: {exc}"])

    # Check artifact hashes
    for entry in manifest.get("artifacts", []):
        artifact_path = run_dir / entry["path"]
        if not artifact_path.exists():
            errors.append(f"Artifact listed in manifest but missing: {entry['path']}")
            continue
        actual_hash = hash_file(artifact_path)
        if actual_hash != entry["sha256"]:
            errors.append(
                f"Hash mismatch for {entry['path']}: "
                f"manifest={entry['sha256'][:16]}... actual={actual_hash[:16]}..."
            )
        actual_size = artifact_path.stat().st_size
        if actual_size != entry["size"]:
            errors.append(
                f"Size mismatch for {entry['path']}: "
                f"manifest={entry['size']} actual={actual_size}"
            )

    # Check metrics for NaN/Infinity
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            _check_finite_recursive(metrics, errors, "metrics.json")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"Invalid metrics.json: {exc}")

    # Fixture cannot be labelled reviewed
    if manifest.get("fixture", True) and manifest.get("reviewed", False):
        errors.append("Fixture run cannot be labelled as reviewed")

    return ValidationResult(valid=not errors, errors=errors)


def validate_summary(summary: dict[str, Any]) -> ValidationResult:
    """Validate a summary dict (multi-seed, repeated-eval, CV, or comparison).

    Checks:
    - No NaN/Infinity in any nested numeric values
    - JSON-serializable with allow_nan=False
    - Identifies the path to any non-finite value
    """
    errors: list[str] = []
    _check_finite_recursive(summary, errors, "summary")

    # Verify strict JSON serialization (no NaN literals)
    try:
        json.dumps(summary, allow_nan=False)
    except ValueError as exc:
        errors.append(f"Summary not strict-JSON-serializable: {exc}")

    return ValidationResult(valid=not errors, errors=errors)


def validate_summary_file(path: Path) -> ValidationResult:
    """Validate a summary.json file.

    Checks:
    - File exists and is valid JSON
    - No NaN/Infinity in any nested numeric values
    - Parses with strict JSON (no NaN literals)
    """
    if not path.exists():
        return ValidationResult(valid=False, errors=[f"File not found: {path}"])

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ValidationResult(valid=False, errors=[f"Cannot read {path}: {exc}"])

    # Strict parse: reject NaN/Infinity literals
    try:
        data = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON literal: {x}")
        ))
    except (json.JSONDecodeError, ValueError) as exc:
        return ValidationResult(valid=False, errors=[f"Invalid JSON in {path}: {exc}"])

    return validate_summary(data)


def _check_finite_recursive(obj: Any, errors: list[str], path: str) -> None:
    """Recursively check all numeric values are finite."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite_recursive(v, errors, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_finite_recursive(v, errors, f"{path}[{i}]")
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            errors.append(f"Non-finite value at {path}: {obj}")
