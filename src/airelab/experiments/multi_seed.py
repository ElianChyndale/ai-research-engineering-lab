"""Multi-seed experiment execution and aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from airelab.core.aggregation import aggregate_metrics
from airelab.core.config import ExperimentConfig
from airelab.experiments.runner import run_experiment


@dataclass(frozen=True)
class MultiSeedConfig:
    """Configuration for running an experiment with multiple seeds."""

    base_config: ExperimentConfig
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = ""

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        for seed in self.seeds:
            if seed < 0:
                raise ValueError(f"Seed must be non-negative, got {seed}")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must not contain duplicates")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if ".." in self.output_dir:
            raise ValueError(f"Path traversal detected in output_dir: {self.output_dir!r}")


def run_multi_seed(config: MultiSeedConfig) -> dict[str, Any]:
    """Run an experiment with multiple seeds and aggregate results.

    Creates subdirectories seed_<n>/ for each seed, then writes summary.json
    with aggregated metrics across all seeds.
    """
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics: list[dict[str, Any]] = []
    per_seed_dirs: list[str] = []

    for seed in config.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_config = ExperimentConfig(
            experiment_id=f"{config.base_config.experiment_id}_seed{seed}",
            experiment_type=config.base_config.experiment_type,
            seed=seed,
            dataset_id=config.base_config.dataset_id,
            parameters=dict(config.base_config.parameters),
            output_dir=str(seed_dir),
            notes=config.base_config.notes,
            fixture=config.base_config.fixture,
        )
        run_experiment(seed_config)

        metrics_path = seed_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_seed_metrics.append(metrics)
        per_seed_dirs.append(str(seed_dir))

    aggregated = aggregate_metrics(per_seed_metrics)

    summary = {
        "experiment_id": config.base_config.experiment_id,
        "experiment_type": config.base_config.experiment_type.value,
        "seeds": list(config.seeds),
        "n_seeds": len(config.seeds),
        "per_seed_dirs": per_seed_dirs,
        "per_seed_metrics": per_seed_metrics,
        "aggregated_metrics": aggregated,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    return summary
