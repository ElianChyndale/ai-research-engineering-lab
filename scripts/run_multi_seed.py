"""CLI entry point for running experiments with multiple seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import yaml  # noqa: E402

from airelab.core.config import ExperimentConfig, ExperimentType  # noqa: E402
from airelab.experiments.multi_seed import MultiSeedConfig, run_multi_seed  # noqa: E402


def load_multi_seed_config(config_path: Path) -> MultiSeedConfig:
    """Load multi-seed config from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = data["base_config"]
    base_config = ExperimentConfig(
        experiment_id=base["experiment_id"],
        experiment_type=ExperimentType(base["experiment_type"]),
        seed=int(base.get("seed", 0)),
        dataset_id=base["dataset_id"],
        parameters=base.get("parameters", {}),
        output_dir="",  # overridden by multi-seed
        notes=base.get("notes", ""),
        fixture=base.get("fixture", True),
    )

    return MultiSeedConfig(
        base_config=base_config,
        seeds=data["seeds"],
        output_dir=data["output_dir"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment with multiple seeds")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_multi_seed_config(config_path)

    # Import all foundation modules to register experiments
    import airelab.foundations.bm25  # noqa: F401
    import airelab.foundations.linear_regression  # noqa: F401
    import airelab.foundations.logistic_regression  # noqa: F401
    import airelab.foundations.pca  # noqa: F401

    summary = run_multi_seed(config)
    print(f"Multi-seed experiment complete: {config.output_dir}")
    print(f"Seeds: {summary['seeds']}")


if __name__ == "__main__":
    main()
