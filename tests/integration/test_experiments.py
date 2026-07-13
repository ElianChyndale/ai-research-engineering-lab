"""Integration tests for experiment runner and validation."""

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
class TestExperiments:
    def test_linear_regression_experiment(self, tmp_path: Path) -> None:
        cfg = {
            "experiment_id": "lr-test",
            "experiment_type": "linear_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 50, "n_features": 2, "noise_std": 0.5},
            "output_dir": str(tmp_path / "lr-test"),
            "fixture": True,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr
        run_dir = tmp_path / "lr-test"
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.csv").exists()
        assert (run_dir / "manifest.json").exists()

    def test_logistic_regression_experiment(self, tmp_path: Path) -> None:
        cfg = {
            "experiment_id": "logr-test",
            "experiment_type": "logistic_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 50, "n_features": 2},
            "output_dir": str(tmp_path / "logr-test"),
            "fixture": True,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

    def test_bm25_experiment(self, tmp_path: Path) -> None:
        cfg = {
            "experiment_id": "bm25-test",
            "experiment_type": "bm25",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"k1": 1.2, "b": 0.75},
            "output_dir": str(tmp_path / "bm25-test"),
            "fixture": True,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        result = run_script([
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config", str(cfg_path),
        ])
        assert result.returncode == 0, result.stderr

    def test_deterministic_rerun(self, tmp_path: Path) -> None:
        """Same config + seed produces identical artifacts."""
        cfg = {
            "experiment_id": "det-test",
            "experiment_type": "linear_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 50, "n_features": 2},
            "output_dir": str(tmp_path / "run1"),
            "fixture": True,
        }
        cfg1 = tmp_path / "config1.yaml"
        cfg1.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )

        cfg["output_dir"] = str(tmp_path / "run2")
        cfg2 = tmp_path / "config2.yaml"
        cfg2.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )

        run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg1)])
        run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg2)])

        # Compare metrics
        m1 = json.loads((tmp_path / "run1" / "metrics.json").read_text(encoding="utf-8"))
        m2 = json.loads((tmp_path / "run2" / "metrics.json").read_text(encoding="utf-8"))
        assert m1 == m2

    def test_validate_valid_run(self, tmp_path: Path) -> None:
        cfg = {
            "experiment_id": "val-test",
            "experiment_type": "linear_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 30, "n_features": 2},
            "output_dir": str(tmp_path / "val-test"),
            "fixture": True,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg_path)])
        result = run_script([
            str(ROOT / "scripts" / "validate_artifacts.py"),
            "--run-dir", str(tmp_path / "val-test"),
        ])
        assert result.returncode == 0, result.stderr
        assert "VALID" in result.stdout

    def test_validate_tampered_run_fails(self, tmp_path: Path) -> None:
        cfg = {
            "experiment_id": "tamper-test",
            "experiment_type": "linear_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 30, "n_features": 2},
            "output_dir": str(tmp_path / "tamper-test"),
            "fixture": True,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )
        run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg_path)])

        # Tamper with metrics
        metrics_path = tmp_path / "tamper-test" / "metrics.json"
        metrics_path.write_text('{"tampered": true}\n', encoding="utf-8")

        result = run_script([
            str(ROOT / "scripts" / "validate_artifacts.py"),
            "--run-dir", str(tmp_path / "tamper-test"),
        ])
        assert result.returncode == 1

    def test_manifest_overwrite_on_rerun(self, tmp_path: Path) -> None:
        """Running an experiment twice into the same directory must not fail."""
        cfg = {
            "experiment_id": "overwrite-test",
            "experiment_type": "linear_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 30, "n_features": 2},
            "output_dir": str(tmp_path / "overwrite-test"),
            "fixture": True,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
            encoding="utf-8",
        )

        # First run
        r1 = run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg_path)])
        assert r1.returncode == 0, r1.stderr

        # Second run into same directory — must not raise FileExistsError
        r2 = run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg_path)])
        assert r2.returncode == 0, r2.stderr

        # Manifest is valid JSON and contains the latest run
        manifest = json.loads(
            (tmp_path / "overwrite-test" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["experiment_id"] == "overwrite-test"
        assert manifest["success"] is True

        # Artifact validation still passes
        val = run_script([
            str(ROOT / "scripts" / "validate_artifacts.py"),
            "--run-dir", str(tmp_path / "overwrite-test"),
        ])
        assert val.returncode == 0, val.stderr
        assert "VALID" in val.stdout

    def test_compare_runs(self, tmp_path: Path) -> None:
        cfg_base = {
            "experiment_type": "linear_regression",
            "seed": 42,
            "dataset_id": "synthetic",
            "parameters": {"n_samples": 30, "n_features": 2},
            "fixture": True,
        }
        cfg1 = {**cfg_base, "experiment_id": "cmp-a", "output_dir": str(tmp_path / "cmp-a")}
        cfg2 = {**cfg_base, "experiment_id": "cmp-b", "output_dir": str(tmp_path / "cmp-b")}

        for cfg in (cfg1, cfg2):
            p = tmp_path / f"config_{cfg['experiment_id']}.yaml"
            p.write_text(
                "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
                encoding="utf-8",
            )
            run_script([str(ROOT / "scripts" / "run_experiment.py"), "--config", str(p)])

        result = run_script([
            str(ROOT / "scripts" / "compare_runs.py"),
            str(tmp_path / "cmp-a"),
            str(tmp_path / "cmp-b"),
        ])
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["same_experiment_type"] is True
        assert data["comparison_meaningful"] is True
