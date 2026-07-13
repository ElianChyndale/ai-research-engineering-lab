"""Tests for airelab.core.artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from airelab.core.artifacts import ArtifactHash, hash_bytes, hash_file


@pytest.mark.unit
class TestArtifacts:
    def test_hash_bytes_deterministic(self) -> None:
        data = b"hello world"
        a = hash_bytes(data)
        b = hash_bytes(data)
        assert a == b

    def test_hash_bytes_different_data(self) -> None:
        a = hash_bytes(b"hello")
        b = hash_bytes(b"world")
        assert a != b

    def test_hash_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        h = hash_file(f)
        assert h == hash_bytes(b"hello world")

    def test_hash_file_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "missing.txt")

    def test_artifact_hash_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"key": "value"}))
        ah = ArtifactHash.from_file(f)
        assert ah.path == f
        assert len(ah.sha256) == 64  # hex digest
        assert ah.size > 0
