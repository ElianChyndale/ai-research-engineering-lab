"""CLI entry point for repeated train/test evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import yaml  # noqa: E402

from airelab.core.config import ExperimentConfig, ExperimentType  # noqa: E402
from airelab.experiments.repeated_eval import RepeatedEvalConfig, run_repeated_eval  # noqa: E402


def load_repeated_eval_config(config_path: Path) -> RepeatedEvalConfig:
    """Load repeated eval config from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = data["base_config"]
    base_config = ExperimentConfig(
        experiment_id=base["experiment_id"],
        experiment_type=ExperimentType(base["experiment_type"]),
        seed=int(base.get("seed", 0)),
        dataset_id=base["dataset_id"],
        parameters=base.get("parameters", {}),
        output_dir="",
        notes=base.get("notes", ""),
        fixture=base.get("fixture", True),
    )

    return RepeatedEvalConfig(
        base_config=base_config,
        n_splits=data.get("n_splits", 5),
        test_size=data.get("test_size", 0.3),
        seeds=data.get("seeds", [42, 43, 44, 45, 46]),
        output_dir=data["output_dir"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated train/test evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_repeated_eval_config(config_path)

    # Import foundation modules
    import airelab.foundations.linear_regression  # noqa: F401
    import airelab.foundations.logistic_regression  # noqa: F401

    summary = run_repeated_eval(config)
    print(f"Repeated evaluation complete: {config.output_dir}")
    print(f"Splits: {summary['n_splits']}, test_size: {summary['test_size']}")


if __name__ == "__main__":
    main()
