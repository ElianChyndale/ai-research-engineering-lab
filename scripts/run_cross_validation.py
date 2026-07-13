"""CLI entry point for k-fold cross-validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import yaml  # noqa: E402

from airelab.core.config import ExperimentConfig, ExperimentType  # noqa: E402
from airelab.experiments.cross_validation import CrossValidationConfig, run_cross_validation  # noqa: E402


def load_cv_config(config_path: Path) -> CrossValidationConfig:
    """Load cross-validation config from YAML file."""
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

    return CrossValidationConfig(
        base_config=base_config,
        n_folds=data.get("n_folds", 5),
        shuffle=data.get("shuffle", True),
        seed=data.get("seed", 42),
        output_dir=data["output_dir"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_cv_config(config_path)

    # Import foundation modules
    import airelab.foundations.linear_regression  # noqa: F401
    import airelab.foundations.logistic_regression  # noqa: F401

    summary = run_cross_validation(config)
    print(f"Cross-validation complete: {config.output_dir}")
    print(f"Folds: {summary['n_folds']}, seed: {summary['seed']}")


if __name__ == "__main__":
    main()
