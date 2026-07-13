"""Experiment manifest with artifact tracking."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

from airelab.core.artifacts import ArtifactHash
from airelab.core.config import ExperimentConfig
from airelab.core.environment import get_environment


def _git_info() -> tuple[str, bool]:
    """Return (commit hash, is_dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ExperimentManifest:
    """Builds and writes experiment manifests."""

    SCHEMA_VERSION = 1

    def __init__(self, config: ExperimentConfig, command: str = "") -> None:
        self._config = config
        self._command = command
        self._start_time = _now_iso()
        self._end_time: str | None = None
        self._success: bool | None = None
        self._artifacts: list[dict[str, Any]] = []
        self._git_commit, self._git_dirty = _git_info()
        self._env = get_environment()

    def mark_artifact(self, ah: ArtifactHash) -> None:
        self._artifacts.append({
            "path": str(ah.path),
            "sha256": ah.sha256,
            "size": ah.size,
        })

    def finish(self, success: bool) -> None:
        self._end_time = _now_iso()
        self._success = success

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "experiment_id": self._config.experiment_id,
            "experiment_type": self._config.experiment_type.value,
            "seed": self._config.seed,
            "git_commit": self._git_commit,
            "git_dirty": self._git_dirty,
            "command": self._command,
            "configuration": self._config.to_dict(),
            "python_version": self._env["python_version"],
            "dependencies": self._env["dependencies"],
            "start_time": self._start_time,
            "end_time": self._end_time,
            "fixture": self._config.fixture,
            "artifacts": self._artifacts,
            "success": self._success,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        text = json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
        # Atomic write: write to temp then rename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.rename(path)
