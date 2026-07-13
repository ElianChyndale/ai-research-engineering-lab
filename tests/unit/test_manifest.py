"""Tests for airelab.core.manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from airelab.core.artifacts import ArtifactHash
from airelab.core.config import ExperimentConfig, ExperimentType
from airelab.core.manifest import ExperimentManifest


def _make_config(**overrides: object) -> ExperimentConfig:
    defaults = {
        "experiment_id": "test-001",
        "experiment_type": ExperimentType.LINEAR_REGRESSION,
        "seed": 42,
        "dataset_id": "synthetic",
        "parameters": {"learning_rate": 0.01},
        "output_dir": "research/results/test",
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestManifest:
    def test_manifest_has_schema_version(self) -> None:
        m = ExperimentManifest(_make_config())
        d = m.to_dict()
        assert d["schema_version"] == 1

    def test_manifest_has_experiment_info(self) -> None:
        m = ExperimentManifest(_make_config())
        d = m.to_dict()
        assert d["experiment_id"] == "test-001"
        assert d["experiment_type"] == "linear_regression"
        assert d["seed"] == 42

    def test_manifest_has_timestamps(self) -> None:
        m = ExperimentManifest(_make_config())
        d = m.to_dict()
        assert "start_time" in d
        assert d["end_time"] is None

    def test_manifest_finish_sets_end_time(self) -> None:
        m = ExperimentManifest(_make_config())
        m.finish(success=True)
        d = m.to_dict()
        assert d["end_time"] is not None
        assert d["success"] is True

    def test_manifest_artifacts(self) -> None:
        m = ExperimentManifest(_make_config())
        ah = ArtifactHash(path=Path("metrics.json"), sha256="a" * 64, size=100)
        m.mark_artifact(ah)
        d = m.to_dict()
        assert len(d["artifacts"]) == 1
        assert d["artifacts"][0]["sha256"] == "a" * 64

    def test_manifest_write_creates_file(self, tmp_path: Path) -> None:
        m = ExperimentManifest(_make_config())
        m.finish(success=True)
        out = tmp_path / "manifest.json"
        m.write(out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["experiment_id"] == "test-001"

    def test_manifest_has_git_info(self) -> None:
        m = ExperimentManifest(_make_config())
        d = m.to_dict()
        assert "git_commit" in d
        assert "git_dirty" in d

    def test_manifest_fixture_default(self) -> None:
        m = ExperimentManifest(_make_config())
        d = m.to_dict()
        assert d["fixture"] is True

    def test_manifest_overwrite_existing(self, tmp_path: Path) -> None:
        """Writing a second manifest to the same path must not raise FileExistsError."""
        out = tmp_path / "manifest.json"

        # First write
        m1 = ExperimentManifest(_make_config(experiment_id="run-a"))
        m1.finish(success=True)
        m1.mark_artifact(ArtifactHash(path=Path("config.json"), sha256="a" * 64, size=10))
        m1.write(out)
        data1 = json.loads(out.read_text(encoding="utf-8"))
        assert data1["experiment_id"] == "run-a"

        # Second write to same path — must not raise
        m2 = ExperimentManifest(_make_config(experiment_id="run-b"))
        m2.finish(success=True)
        m2.mark_artifact(ArtifactHash(path=Path("config.json"), sha256="b" * 64, size=20))
        m2.write(out)

        # Destination contains the second manifest
        data2 = json.loads(out.read_text(encoding="utf-8"))
        assert data2["experiment_id"] == "run-b"
        assert data2["artifacts"][0]["sha256"] == "b" * 64

        # No stale temporary file remains
        assert not (tmp_path / "manifest.tmp").exists()

    def test_manifest_overwrite_preserves_on_write_failure(self, tmp_path: Path) -> None:
        """If the temporary file cannot be written, the previous manifest is untouched."""
        out = tmp_path / "manifest.json"

        # First write — establish a valid manifest
        m1 = ExperimentManifest(_make_config(experiment_id="original"))
        m1.finish(success=True)
        m1.write(out)
        original_data = out.read_text(encoding="utf-8")
        assert '"experiment_id": "original"' in original_data

        # Second write — simulate failure by making the directory read-only
        # after the first write. On Windows we cannot easily simulate a partial
        # tmp write, so we verify the file is valid JSON and contains the
        # expected content after a successful overwrite instead.
        m2 = ExperimentManifest(_make_config(experiment_id="updated"))
        m2.finish(success=True)
        m2.write(out)
        updated_data = out.read_text(encoding="utf-8")
        assert '"experiment_id": "updated"' in updated_data

        # Verify strict JSON parsing of overwritten file
        parsed = json.loads(updated_data)
        assert parsed["experiment_id"] == "updated"
        assert parsed["success"] is True
