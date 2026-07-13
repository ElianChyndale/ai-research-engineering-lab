"""Experiment runner — orchestrates config → run → manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from airelab.core.artifacts import ArtifactHash
from airelab.core.config import ExperimentConfig
from airelab.core.manifest import ExperimentManifest
from airelab.core.seeds import set_seed
from airelab.experiments.registry import get_experiment


def run_experiment(config: ExperimentConfig) -> Path:
    """Run an experiment and write all artifacts.

    Returns the run directory path.
    """
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = ExperimentManifest(
        config=config,
        command=" ".join(sys.argv),
    )

    # Write config artifact
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Set deterministic seed
    set_seed(config.seed)

    success = False
    try:
        # Run the experiment
        experiment_fn = get_experiment(config.experiment_type.value)
        metrics = experiment_fn(config, run_dir)

        # Write metrics artifact
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        success = True
    except Exception:
        success = False
        raise
    finally:
        # Compute hashes for all artifacts
        manifest.finish(success=success)
        for artifact_file in sorted(run_dir.iterdir()):
            if artifact_file.is_file() and artifact_file.suffix in (".json", ".csv") and artifact_file.name != "manifest.json":
                ah = ArtifactHash.from_file(artifact_file)
                # Store relative path (filename only)
                manifest.mark_artifact(ArtifactHash(
                    path=Path(artifact_file.name),
                    sha256=ah.sha256,
                    size=ah.size,
                ))

        # Write environment.json
        from airelab.core.environment import get_environment

        env_path = run_dir / "environment.json"
        env_path.write_text(
            json.dumps(get_environment(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        # Write manifest last (includes env hash)
        env_hash = ArtifactHash.from_file(env_path)
        manifest.mark_artifact(ArtifactHash(
            path=Path("environment.json"),
            sha256=env_hash.sha256,
            size=env_hash.size,
        ))
        manifest.write(run_dir / "manifest.json")

    return run_dir
