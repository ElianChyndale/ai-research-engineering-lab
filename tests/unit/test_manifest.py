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
        "parameters": {"lr": 0.01},
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
