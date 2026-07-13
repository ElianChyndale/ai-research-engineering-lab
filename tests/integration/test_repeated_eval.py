"""Integration tests for repeated train/test evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable] + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
class TestRepeatedEval:
    def test_repeated_eval_lr(self, tmp_path: Path) -> None:
        """Repeated eval LR produces summary with train/test metrics."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-rep-test",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 50, "n_features": 2, "noise_std": 0.5},
                "fixture": True,
            },
            "n_splits": 3,
            "test_size": 0.3,
            "seeds": [42, 43, 44],
            "output_dir": str(tmp_path / "lr-rep"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_repeated_eval.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

        summary = json.loads(
            (tmp_path / "lr-rep" / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["n_splits"] == 3
        assert summary["test_size"] == 0.3
        assert len(summary["per_split_metrics"]) == 3

        # Must have train and test metrics
        agg = summary["aggregated_metrics"]
        assert "train_mse" in agg
        assert "test_mse" in agg
        assert "train_r2" in agg
        assert "test_r2" in agg

    def test_repeated_eval_logr(self, tmp_path: Path) -> None:
        """Repeated eval LogR produces summary with accuracy metrics."""
        cfg = {
            "base_config": {
                "experiment_id": "logr-rep-test",
                "experiment_type": "logistic_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 50, "n_features": 2},
                "fixture": True,
            },
            "n_splits": 3,
            "test_size": 0.3,
            "seeds": [42, 43, 44],
            "output_dir": str(tmp_path / "logr-rep"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_repeated_eval.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

        summary = json.loads(
            (tmp_path / "logr-rep" / "summary.json").read_text(encoding="utf-8")
        )
        agg = summary["aggregated_metrics"]
        assert "train_accuracy" in agg
        assert "test_accuracy" in agg

    def test_repeated_eval_deterministic(self, tmp_path: Path) -> None:
        """Same repeated eval config produces identical summary."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-det-rep",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2},
                "fixture": True,
            },
            "n_splits": 2,
            "test_size": 0.3,
            "seeds": [42, 43],
            "output_dir": str(tmp_path / "run1"),
        }
        for run_name in ("run1", "run2"):
            cfg["output_dir"] = str(tmp_path / run_name)
            cfg_path = tmp_path / f"config_{run_name}.yaml"
            cfg_path.write_text(
                "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
                encoding="utf-8",
            )
            run_script([
                str(ROOT / "scripts" / "run_repeated_eval.py"),
                "--config", str(cfg_path),
            ])

        s1 = json.loads((tmp_path / "run1" / "summary.json").read_text(encoding="utf-8"))
        s2 = json.loads((tmp_path / "run2" / "summary.json").read_text(encoding="utf-8"))
        assert s1["aggregated_metrics"] == s2["aggregated_metrics"]
