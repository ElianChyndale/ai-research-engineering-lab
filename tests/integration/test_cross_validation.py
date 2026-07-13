"""Integration tests for k-fold cross-validation."""

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
class TestCrossValidation:
    def test_cv_lr(self, tmp_path: Path) -> None:
        """5-fold CV LR produces per-fold metrics and aggregates."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-cv-test",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 50, "n_features": 2, "noise_std": 0.5},
                "fixture": True,
            },
            "n_folds": 5,
            "shuffle": True,
            "seed": 42,
            "output_dir": str(tmp_path / "lr-cv"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_cross_validation.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

        summary = json.loads(
            (tmp_path / "lr-cv" / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["n_folds"] == 5
        assert summary["seed"] == 42
        assert len(summary["per_fold_metrics"]) == 5

        # Each fold must have train/test metrics
        for fold_m in summary["per_fold_metrics"]:
            assert "train_mse" in fold_m
            assert "test_mse" in fold_m
            assert "fold" in fold_m

        # Aggregated metrics
        agg = summary["aggregated_metrics"]
        assert "test_mse" in agg
        assert "mean" in agg["test_mse"]
        assert "std" in agg["test_mse"]

    def test_cv_logr(self, tmp_path: Path) -> None:
        """5-fold CV LogR produces per-fold accuracy metrics."""
        cfg = {
            "base_config": {
                "experiment_id": "logr-cv-test",
                "experiment_type": "logistic_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 50, "n_features": 2},
                "fixture": True,
            },
            "n_folds": 5,
            "shuffle": True,
            "seed": 42,
            "output_dir": str(tmp_path / "logr-cv"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_cross_validation.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

        summary = json.loads(
            (tmp_path / "logr-cv" / "summary.json").read_text(encoding="utf-8")
        )
        agg = summary["aggregated_metrics"]
        assert "test_accuracy" in agg
        assert "train_accuracy" in agg

    def test_cv_deterministic(self, tmp_path: Path) -> None:
        """Same CV config produces identical summary."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-cv-det",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2},
                "fixture": True,
            },
            "n_folds": 3,
            "shuffle": True,
            "seed": 42,
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
                str(ROOT / "scripts" / "run_cross_validation.py"),
                "--config", str(cfg_path),
            ])

        s1 = json.loads((tmp_path / "run1" / "summary.json").read_text(encoding="utf-8"))
        s2 = json.loads((tmp_path / "run2" / "summary.json").read_text(encoding="utf-8"))
        assert s1["aggregated_metrics"] == s2["aggregated_metrics"]

    def test_cv_fold_count(self, tmp_path: Path) -> None:
        """3-fold CV produces exactly 3 folds."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-cv-3fold",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2},
                "fixture": True,
            },
            "n_folds": 3,
            "shuffle": False,
            "seed": 42,
            "output_dir": str(tmp_path / "cv3"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_cross_validation.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

        summary = json.loads(
            (tmp_path / "cv3" / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["n_folds"] == 3
        assert len(summary["per_fold_metrics"]) == 3

        # Total test samples should equal total samples
        total_test = sum(f["n_test"] for f in summary["per_fold_metrics"])
        assert total_test == 30
