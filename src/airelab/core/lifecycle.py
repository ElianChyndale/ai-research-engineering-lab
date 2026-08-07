"""Experiment lifecycle states and config-freeze guard.

Represents EXPLORATORY / FROZEN / CONFIRMATORY / INVALIDATED states and
prevents accidental modification of a frozen confirmatory configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from airelab.core.provenance import sha256_file


class ExperimentState(str, Enum):
    EXPLORATORY = "EXPLORATORY"
    FROZEN = "FROZEN"
    CONFIRMATORY = "CONFIRMATORY"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True)
class FrozenConfig:
    """A frozen config file plus its recorded SHA-256.

    The recorded hash is captured at freeze time; a later change to the file
    changes the hash and is detected as a violation.
    """

    path: Path
    state: ExperimentState
    recorded_sha256: str

    @classmethod
    def freeze(cls, path: Path) -> "FrozenConfig":
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        return cls(path=path, state=ExperimentState.FROZEN, recorded_sha256=sha256_file(path))

    def verify(self) -> bool:
        """True iff the file still hashes to the recorded value."""
        return sha256_file(self.path) == self.recorded_sha256

    def promote_to_confirmatory(self) -> "FrozenConfig":
        if not self.verify():
            raise ValueError(
                f"config changed since freeze ({self.path}); "
                "cannot promote a modified config to confirmatory"
            )
        return FrozenConfig(self.path, ExperimentState.CONFIRMATORY, self.recorded_sha256)

    def invalidate(self) -> "FrozenConfig":
        return FrozenConfig(self.path, ExperimentState.INVALIDATED, self.recorded_sha256)


def hash_matches_record(config: FrozenConfig) -> bool:
    """Convenience predicate (same as verify)."""
    return config.verify()
