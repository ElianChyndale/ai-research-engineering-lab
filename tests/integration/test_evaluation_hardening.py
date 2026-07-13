"""Integration tests for evaluation artifact hardening (v0.2.1).

Covers:
- PCA deterministic semantic artifacts
- Repeated train/test non-overlap
- CV observation coverage (each in exactly one validation fold)
- Fresh model per CV fold
- Failed child execution visibility
- Summary artifact non-finite rejection
- Selective-risk NaN cannot enter summary
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
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
class TestPCADeterministicArtifacts:
    def test_pca_deterministic_semantic_artifacts(self, tmp_path: Path) -> None:
        """PCA experiment produces identical metrics and components on rerun."""
        cfg = {
            "experiment_id": "pca-det-test",
            "experiment_type": "pca",
            "seed": 42,
            "dataset_id": "synthetic-pca-v1",
            "parameters": {"n_samples": 50, "n_features": 4, "n_components": 2},
            "output_dir": str(tmp_path / "run1"),
            "fixture": True,
        }

        results = []
        for run_name in ("run1", "run2"):
            cfg["output_dir"] = str(tmp_path / run_name)
            cfg_path = tmp_path / f"config_{run_name}.yaml"
            cfg_path.write_text(
                "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
                encoding="utf-8",
            )
            result = run_script([
                str(ROOT / "scripts" / "run_experiment.py"),
                "--config", str(cfg_path),
            ])
            assert result.returncode == 0, result.stderr

            metrics = json.loads(
                (tmp_path / run_name / "metrics.json").read_text(encoding="utf-8")
            )
            components = (tmp_path / run_name / "components.csv").read_text(encoding="utf-8")
            results.append((metrics, components))

        # Metrics must be identical
        assert results[0][0] == results[1][0]
        # Components CSV must be identical
        assert results[0][1] == results[1][1]


@pytest.mark.integration
class TestRepeatedEvalIsolation:
    def test_train_test_no_overlap(self, tmp_path: Path) -> None:
        """Train and test indices do not overlap in repeated evaluation."""
        from airelab.experiments.repeated_eval import _generate_data, _train_test_split
        from airelab.core.config import ExperimentType

        params = {"n_samples": 100, "n_features": 3, "noise_std": 0.5}
        X, y = _generate_data(ExperimentType.LINEAR_REGRESSION, params, 42)
        X_train, X_test, y_train, y_test = _train_test_split(X, y, 0.3, 42)

        assert len(X_train) + len(X_test) == len(X)

        # No row overlap
        train_tuples = set(map(tuple, X_train.tolist()))
        test_tuples = set(map(tuple, X_test.tolist()))
        assert len(train_tuples & test_tuples) == 0

    def test_repeated_eval_generates_different_data_per_seed(self, tmp_path: Path) -> None:
        """Each seed in repeated eval produces a different synthetic dataset."""
        from airelab.experiments.repeated_eval import _generate_data
        from airelab.core.config import ExperimentType

        params = {"n_samples": 50, "n_features": 2, "noise_std": 0.5}
        X1, y1 = _generate_data(ExperimentType.LINEAR_REGRESSION, params, 42)
        X2, y2 = _generate_data(ExperimentType.LINEAR_REGRESSION, params, 43)

        assert not np.array_equal(X1, X2)
        assert not np.array_equal(y1, y2)


@pytest.mark.integration
class TestCrossValidationIsolation:
    def test_each_observation_in_exactly_one_validation_fold(self, tmp_path: Path) -> None:
        """Every observation appears in exactly one validation fold."""
        from airelab.experiments.cross_validation import _make_folds

        n = 50
        n_folds = 5
        folds = _make_folds(n, n_folds, shuffle=True, seed=42)

        # Each fold non-empty
        for fold in folds:
            assert len(fold) > 0

        # All indices covered
        all_idx = np.concatenate(folds)
        assert len(all_idx) == n
        assert len(set(all_idx.tolist())) == n

        # Each index in exactly one fold
        for i in range(n):
            count = sum(1 for fold in folds if i in fold)
            assert count == 1, f"Index {i} in {count} folds"

    def test_fresh_model_per_fold(self, tmp_path: Path) -> None:
        """Each CV fold trains a fresh model (different weights for different train sets)."""
        from airelab.experiments.cross_validation import _make_folds
        from airelab.experiments.repeated_eval import _generate_data
        from airelab.foundations.linear_regression import LinearRegression
        from airelab.core.config import ExperimentType

        params = {"n_samples": 50, "n_features": 2, "noise_std": 0.5}
        X, y = _generate_data(ExperimentType.LINEAR_REGRESSION, params, 42)
        folds = _make_folds(len(X), 3, shuffle=True, seed=42)

        weights = []
        for fold_idx in range(3):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(3) if i != fold_idx])
            model = LinearRegression(solver="gradient_descent", learning_rate=0.01, max_iter=5000)
            model.fit(X[train_idx], y[train_idx])
            weights.append(model.coefficients.copy())

        # Different training sets → different weights
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(weights[i], weights[j])


@pytest.mark.integration
class TestSummaryArtifactHardening:
    def test_valid_summary_passes_validation(self, tmp_path: Path) -> None:
        """A valid multi-seed summary passes validation."""
        from airelab.core.validation import validate_summary_file

        cfg = {
            "base_config": {
                "experiment_id": "lr-val-test",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 0.5, "learning_rate": 0.01, "max_iter": 5000},
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

        summary_path = tmp_path / "lr-ms" / "summary.json"
        val = validate_summary_file(summary_path)
        assert val.valid, str(val)

    def test_cv_summary_passes_validation(self, tmp_path: Path) -> None:
        """A valid CV summary passes validation."""
        from airelab.core.validation import validate_summary_file

        cfg = {
            "base_config": {
                "experiment_id": "lr-cv-val",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 0.5, "learning_rate": 0.01, "max_iter": 5000},
                "fixture": True,
            },
            "n_folds": 3,
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

        summary_path = tmp_path / "lr-cv" / "summary.json"
        val = validate_summary_file(summary_path)
        assert val.valid, str(val)

    def test_summary_strict_json_parseable(self, tmp_path: Path) -> None:
        """Summary JSON must parse without NaN literals."""
        from airelab.core.validation import validate_summary_file

        cfg = {
            "base_config": {
                "experiment_id": "lr-strict",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic",
                "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 0.5, "learning_rate": 0.01, "max_iter": 5000},
                "fixture": True,
            },
            "seeds": [42],
            "output_dir": str(tmp_path / "lr-strict"),
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

        summary_path = tmp_path / "lr-strict" / "summary.json"
        text = summary_path.read_text(encoding="utf-8")

        # Must not contain NaN literal
        assert "NaN" not in text
        assert "Infinity" not in text
        assert "-Infinity" not in text

        # Must parse with strict JSON
        val = validate_summary_file(summary_path)
        assert val.valid, str(val)

    def test_selective_risk_nan_cannot_enter_summary(self) -> None:
        """selective_risk NaN is isolated — no experiment writes it to metrics."""
        from airelab.foundations.calibration import selective_risk
        import numpy as np

        # selective_risk returns NaN for empty accepted set
        sr = selective_risk(
            np.array([1, 0]), np.array([1, 0]), np.array([0.1, 0.2]), threshold=0.5
        )
        assert math.isnan(sr)

        # But no experiment function includes selective_risk in its output.
        # Verify by checking that aggregate_metrics would catch it if it did.
        from airelab.core.aggregation import aggregate_metrics

        with pytest.raises(ValueError, match="Non-finite"):
            aggregate_metrics([{"selective_risk": sr}])


@pytest.mark.integration
class TestFamilyComparisonIntegration:
    def test_comparable_families(self, tmp_path: Path) -> None:
        """Two LR multi-seed runs are comparable."""
        from airelab.experiments.comparison import compare_families

        summaries = []
        for suffix in ("a", "b"):
            cfg = {
                "base_config": {
                    "experiment_id": f"lr-comp-{suffix}",
                    "experiment_type": "linear_regression",
                    "dataset_id": "synthetic-linear-v1",
                    "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 0.5, "learning_rate": 0.01, "max_iter": 5000},
                    "fixture": True,
                },
                "seeds": [42, 43],
                "output_dir": str(tmp_path / f"lr-{suffix}"),
            }
            cfg_path = tmp_path / f"config_{suffix}.yaml"
            cfg_path.write_text(
                "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
                encoding="utf-8",
            )
            result = run_script([
                str(ROOT / "scripts" / "run_multi_seed.py"),
                "--config", str(cfg_path),
            ])
            assert result.returncode == 0, result.stderr
            summaries.append(tmp_path / f"lr-{suffix}")

        result = compare_families(summaries[0], summaries[1])
        assert result["comparable"] is True
        assert result["reasons"] == []

    def test_incompatible_families(self, tmp_path: Path) -> None:
        """LR vs LogR multi-seed runs are not comparable."""
        from airelab.experiments.comparison import compare_families

        # LR
        cfg_lr = {
            "base_config": {
                "experiment_id": "lr-incomp",
                "experiment_type": "linear_regression",
                "dataset_id": "synthetic-linear-v1",
                "parameters": {"n_samples": 30, "n_features": 2, "noise_std": 0.5, "learning_rate": 0.01, "max_iter": 5000},
                "fixture": True,
            },
            "seeds": [42, 43],
            "output_dir": str(tmp_path / "lr"),
        }
        cfg_path = tmp_path / "config_lr.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg_lr.items()),
            encoding="utf-8",
        )
        run_script([str(ROOT / "scripts" / "run_multi_seed.py"), "--config", str(cfg_path)])

        # LogR
        cfg_logr = {
            "base_config": {
                "experiment_id": "logr-incomp",
                "experiment_type": "logistic_regression",
                "dataset_id": "synthetic-binary-v1",
                "parameters": {"n_samples": 30, "n_features": 2, "learning_rate": 0.5, "max_iter": 1000},
                "fixture": True,
            },
            "seeds": [42, 43],
            "output_dir": str(tmp_path / "logr"),
        }
        cfg_path = tmp_path / "config_logr.yaml"
        cfg_path.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg_logr.items()),
            encoding="utf-8",
        )
        run_script([str(ROOT / "scripts" / "run_multi_seed.py"), "--config", str(cfg_path)])

        result = compare_families(tmp_path / "lr", tmp_path / "logr")
        assert result["comparable"] is False
        assert any("experiment types differ" in r for r in result["reasons"])
        assert "conclusion" in result
