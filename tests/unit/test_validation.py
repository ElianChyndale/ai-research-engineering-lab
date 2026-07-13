"""Tests for airelab.core.validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from airelab.core.validation import ValidationResult, validate_run, validate_summary, validate_summary_file


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_valid_run(run_dir: Path) -> None:
    """Create a minimal valid run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", {"experiment_id": "test", "seed": 42})
    _write_json(run_dir / "metrics.json", {"mse": 0.5, "r2": 0.9})

    # Build manifest with correct hashes
    from airelab.core.artifacts import hash_file

    artifacts = []
    for name in ("config.json", "metrics.json"):
        p = run_dir / name
        h = hash_file(p)
        artifacts.append({"path": name, "sha256": h, "size": p.stat().st_size})

    manifest = {
        "schema_version": 1,
        "experiment_id": "test",
        "seed": 42,
        "fixture": True,
        "artifacts": artifacts,
    }
    _write_json(run_dir / "manifest.json", manifest)


@pytest.mark.unit
class TestValidation:
    def test_valid_run_passes(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-001"
        _make_valid_run(run_dir)
        result = validate_run(run_dir)
        assert result.valid, str(result)

    def test_missing_directory_fails(self, tmp_path: Path) -> None:
        result = validate_run(tmp_path / "nonexistent")
        assert not result.valid

    def test_missing_config_json_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_json(run_dir / "metrics.json", {"mse": 0.5})
        _write_json(run_dir / "manifest.json", {"artifacts": []})
        result = validate_run(run_dir)
        assert not result.valid
        assert any("config.json" in e for e in result.errors)

    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_json(run_dir / "config.json", {})
        _write_json(run_dir / "metrics.json", {})
        result = validate_run(run_dir)
        assert not result.valid
        assert any("manifest.json" in e for e in result.errors)

    def test_hash_mismatch_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _make_valid_run(run_dir)
        # Tamper with config.json
        _write_json(run_dir / "config.json", {"experiment_id": "tampered"})
        result = validate_run(run_dir)
        assert not result.valid
        assert any("Hash mismatch" in e for e in result.errors)

    def test_nan_metric_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _make_valid_run(run_dir)
        _write_json(run_dir / "metrics.json", {"mse": float("nan")})
        # Re-hash metrics in manifest
        from airelab.core.artifacts import hash_file

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"][1] = {
            "path": "metrics.json",
            "sha256": hash_file(run_dir / "metrics.json"),
            "size": (run_dir / "metrics.json").stat().st_size,
        }
        _write_json(run_dir / "manifest.json", manifest)
        result = validate_run(run_dir)
        assert not result.valid
        assert any("Non-finite" in e for e in result.errors)

    def test_inf_metric_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _make_valid_run(run_dir)
        _write_json(run_dir / "metrics.json", {"mse": float("inf")})
        from airelab.core.artifacts import hash_file

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"][1] = {
            "path": "metrics.json",
            "sha256": hash_file(run_dir / "metrics.json"),
            "size": (run_dir / "metrics.json").stat().st_size,
        }
        _write_json(run_dir / "manifest.json", manifest)
        result = validate_run(run_dir)
        assert not result.valid

    def test_fixture_cannot_be_reviewed(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _make_valid_run(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["reviewed"] = True
        _write_json(run_dir / "manifest.json", manifest)
        result = validate_run(run_dir)
        assert not result.valid
        assert any("reviewed" in e.lower() for e in result.errors)

    def test_validation_result_str_valid(self) -> None:
        r = ValidationResult(valid=True, errors=[])
        assert str(r) == "VALID"

    def test_validation_result_str_invalid(self) -> None:
        r = ValidationResult(valid=False, errors=["error1", "error2"])
        assert "INVALID" in str(r)
        assert "error1" in str(r)


@pytest.mark.unit
class TestSummaryValidation:
    def test_valid_summary_passes(self) -> None:
        summary = {
            "experiment_type": "linear_regression",
            "aggregated_metrics": {
                "mse": {"mean": 0.5, "std": 0.1, "median": 0.5, "min": 0.4, "max": 0.6}
            },
        }
        result = validate_summary(summary)
        assert result.valid, str(result)

    def test_nested_nan_fails(self) -> None:
        summary = {
            "aggregated_metrics": {
                "mse": {"mean": float("nan"), "std": 0.1}
            }
        }
        result = validate_summary(summary)
        assert not result.valid
        assert any("Non-finite" in e for e in result.errors)

    def test_nested_inf_fails(self) -> None:
        summary = {
            "per_fold_metrics": [
                {"test_mse": float("inf")}
            ]
        }
        result = validate_summary(summary)
        assert not result.valid
        assert any("Non-finite" in e for e in result.errors)

    def test_deeply_nested_nan_fails(self) -> None:
        summary = {
            "level1": {
                "level2": {
                    "level3": [float("nan")]
                }
            }
        }
        result = validate_summary(summary)
        assert not result.valid
        assert any("level1.level2.level3[0]" in e for e in result.errors)

    def test_strict_json_serialization(self) -> None:
        """Valid summary must serialize with allow_nan=False."""
        summary = {"mse": {"mean": 0.5, "std": 0.1}}
        result = validate_summary(summary)
        assert result.valid
        # Verify it actually serializes strictly
        json.dumps(summary, allow_nan=False)  # should not raise

    def test_summary_file_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        summary = {"mse": {"mean": 0.5}}
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = validate_summary_file(path)
        assert result.valid, str(result)

    def test_summary_file_nan_literal_fails(self, tmp_path: Path) -> None:
        """A file containing JSON NaN literal must fail."""
        path = tmp_path / "summary.json"
        path.write_text('{"mse": NaN}', encoding="utf-8")
        result = validate_summary_file(path)
        assert not result.valid
        assert any("Invalid JSON" in e or "Non-finite" in e for e in result.errors)

    def test_summary_file_missing_fails(self, tmp_path: Path) -> None:
        result = validate_summary_file(tmp_path / "nonexistent.json")
        assert not result.valid
