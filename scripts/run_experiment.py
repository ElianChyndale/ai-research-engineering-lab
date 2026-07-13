"""CLI entry point for running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import yaml  # noqa: E402

from airelab.core.config import ExperimentConfig, ExperimentType  # noqa: E402
from airelab.experiments.runner import run_experiment  # noqa: E402


def load_config(config_path: Path) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(
        experiment_id=data["experiment_id"],
        experiment_type=ExperimentType(data["experiment_type"]),
        seed=int(data["seed"]),
        dataset_id=data["dataset_id"],
        parameters=data.get("parameters", {}),
        output_dir=data.get("output_dir", ""),
        notes=data.get("notes", ""),
        fixture=data.get("fixture", True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)

    # Import all foundation modules to register experiments
    import airelab.foundations.bm25  # noqa: F401
    import airelab.foundations.linear_regression  # noqa: F401
    import airelab.foundations.logistic_regression  # noqa: F401
    import airelab.foundations.pca  # noqa: F401

    run_dir = run_experiment(config)
    print(f"Experiment complete: {run_dir}")


if __name__ == "__main__":
    main()
