"""Integration tests for multi-seed experiment execution."""

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
class TestMultiSeed:
    def test_multi_seed_lr(self, tmp_path: Path) -> None:
        """Multi-seed LR produces per-seed dirs and summary."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-ms-test",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 0.5},
                "fixture": True,
            },
            "seeds": [42, 43],
            "output_dir": str(tmp_path / "lr-ms"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_multi_seed.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

        out_dir = tmp_path / "lr-ms"
        assert (out_dir / "seed_42").is_dir()
        assert (out_dir / "seed_43").is_dir()
        assert (out_dir / "summary.json").exists()

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["seeds"] == [42, 43]
        assert summary["n_seeds"] == 2
        assert len(summary["per_seed_metrics"]) == 2

        # Aggregated metrics should have mean, std, etc.
        agg = summary["aggregated_metrics"]
        assert "ols_mse" in agg
        assert "mean" in agg["ols_mse"]
        assert "std" in agg["ols_mse"]

    def test_multi_seed_deterministic(self, tmp_path: Path) -> None:
        """Same multi-seed config produces identical summary."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-det-test",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2},
                "fixture": True,
            },
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
            run_script([str(ROOT / "scripts" / "run_multi_seed.py"), "--config", str(cfg_path)])

        s1 = json.loads((tmp_path / "run1" / "summary.json").read_text(encoding="utf-8"))
        s2 = json.loads((tmp_path / "run2" / "summary.json").read_text(encoding="utf-8"))
        assert s1["aggregated_metrics"] == s2["aggregated_metrics"]

    def test_multi_seed_variance_visible(self, tmp_path: Path) -> None:
        """Different seeds produce different per-seed metrics."""
        cfg = {
            "base_config": {
                "experiment_id": "lr-var-test",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 1.0},
                "fixture": True,
            },
            "seeds": [42, 99],
            "output_dir": str(tmp_path / "lr-var"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        run_script([str(ROOT / "scripts" / "run_multi_seed.py"), "--config", str(cfg_path)])

        summary = json.loads(
            (tmp_path / "lr-var" / "summary.json").read_text(encoding="utf-8")
        )
        m0 = summary["per_seed_metrics"][0]["ols_r2"]
        m1 = summary["per_seed_metrics"][1]["ols_r2"]
        # With noise_std=1.0 and different seeds, R² should differ
        assert m0 != m1
