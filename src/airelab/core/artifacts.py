"""Artifact hashing with SHA-256."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


def hash_bytes(data: bytes) -> str:
    """Return hex SHA-256 of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    """Return hex SHA-256 of a file's contents."""
    return hash_bytes(path.read_bytes())


@dataclass(frozen=True)
class ArtifactHash:
    path: Path
    sha256: str
    size: int

    @classmethod
    def from_file(cls, path: Path) -> ArtifactHash:
        """Compute hash and size for a file."""
        data = path.read_bytes()
        return cls(path=path, sha256=hash_bytes(data), size=len(data))
