"""Typed experiment configuration with validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentType(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    PCA = "pca"
    BM25 = "bm25"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    experiment_type: ExperimentType
    seed: int
    dataset_id: str
    parameters: dict[str, float] = field(default_factory=dict)
    output_dir: str = ""
    notes: str = ""
    fixture: bool = True

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.experiment_type, ExperimentType):
            raise ValueError(
                f"Unknown experiment type: {self.experiment_type!r}. "
                f"Valid types: {[t.value for t in ExperimentType]}"
            )
        # Validate parameters are finite
        for key, value in self.parameters.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Parameter '{key}' must be finite, got {value!r}")
        # Path traversal check
        if ".." in self.output_dir:
            raise ValueError(f"Path traversal detected in output_dir: {self.output_dir!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type.value,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "parameters": dict(self.parameters),
            "output_dir": self.output_dir,
            "notes": self.notes,
            "fixture": self.fixture,
        }
