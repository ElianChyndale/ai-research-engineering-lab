"""Tests for airelab.experiments.comparison — family comparability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_summary(path: Path, data: dict) -> None:
    (path / "summary.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.unit
class TestFamilyComparability:
    def test_same_family_comparable(self, tmp_path: Path) -> None:
        """Same experiment type, dataset, and protocol → comparable."""
        from airelab.experiments.comparison import compare_families

        d_a = tmp_path / "a"
        d_b = tmp_path / "b"
        d_a.mkdir()
        d_b.mkdir()
        summary = {
            "experiment_type": "linear_regression",
            "dataset_id": "synthetic-linear-v1",
            "fixture": True,
            "seeds": [42, 43],
            "aggregated_metrics": {
                "test_mse": {"mean": 0.2, "std": 0.05, "min": 0.15, "max": 0.25}
            },
        }
        _write_summary(d_a, summary)
        _write_summary(d_b, {**summary, "aggregated_metrics": {
            "test_mse": {"mean": 0.3, "std": 0.04, "min": 0.26, "max": 0.34}
        }})
        result = compare_families(d_a, d_b)
        assert result["comparable"] is True
        assert result["reasons"] == []
        # mean_diff only present when comparable
        assert "mean_diff" in result["metric_comparisons"]["test_mse"]

    def test_different_types_not_comparable(self, tmp_path: Path) -> None:
        from airelab.experiments.comparison import compare_families

        d_a = tmp_path / "a"
        d_b = tmp_path / "b"
        d_a.mkdir()
        d_b.mkdir()
        _write_summary(d_a, {
            "experiment_type": "linear_regression",
            "dataset_id": "synthetic",
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.2, "std": 0.05, "min": 0.15, "max": 0.25}},
        })
        _write_summary(d_b, {
            "experiment_type": "logistic_regression",
            "dataset_id": "synthetic",
            "seeds": [42],
            "aggregated_metrics": {"accuracy": {"mean": 0.95, "std": 0.02, "min": 0.93, "max": 0.97}},
        })
        result = compare_families(d_a, d_b)
        assert result["comparable"] is False
        assert any("experiment types differ" in r for r in result["reasons"])
        assert "conclusion" in result
        assert "No performance conclusion" in result["conclusion"]
        # mean_diff should NOT be present when not comparable
        for v in result["metric_comparisons"].values():
            assert "mean_diff" not in v

    def test_different_datasets_not_comparable(self, tmp_path: Path) -> None:
        from airelab.experiments.comparison import compare_families

        d_a = tmp_path / "a"
        d_b = tmp_path / "b"
        d_a.mkdir()
        d_b.mkdir()
        _write_summary(d_a, {
            "experiment_type": "linear_regression",
            "dataset_id": "dataset-v1",
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.2, "std": 0.05, "min": 0.15, "max": 0.25}},
        })
        _write_summary(d_b, {
            "experiment_type": "linear_regression",
            "dataset_id": "dataset-v2",
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.3, "std": 0.04, "min": 0.26, "max": 0.34}},
        })
        result = compare_families(d_a, d_b)
        assert result["comparable"] is False
        assert any("datasets differ" in r for r in result["reasons"])

    def test_different_protocols_not_comparable(self, tmp_path: Path) -> None:
        from airelab.experiments.comparison import compare_families

        d_a = tmp_path / "a"
        d_b = tmp_path / "b"
        d_a.mkdir()
        d_b.mkdir()
        _write_summary(d_a, {
            "experiment_type": "linear_regression",
            "dataset_id": "synthetic",
            "n_folds": 5,
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.2, "std": 0.05, "min": 0.15, "max": 0.25}},
        })
        _write_summary(d_b, {
            "experiment_type": "linear_regression",
            "dataset_id": "synthetic",
            "n_folds": 10,
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.3, "std": 0.04, "min": 0.26, "max": 0.34}},
        })
        result = compare_families(d_a, d_b)
        assert result["comparable"] is False
        assert any("n_folds" in r for r in result["reasons"])

    def test_different_fixture_status_not_comparable(self, tmp_path: Path) -> None:
        from airelab.experiments.comparison import compare_families

        d_a = tmp_path / "a"
        d_b = tmp_path / "b"
        d_a.mkdir()
        d_b.mkdir()
        _write_summary(d_a, {
            "experiment_type": "linear_regression",
            "dataset_id": "synthetic",
            "fixture": True,
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.2, "std": 0.05, "min": 0.15, "max": 0.25}},
        })
        _write_summary(d_b, {
            "experiment_type": "linear_regression",
            "dataset_id": "synthetic",
            "fixture": False,
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.3, "std": 0.04, "min": 0.26, "max": 0.34}},
        })
        result = compare_families(d_a, d_b)
        assert result["comparable"] is False
        assert any("fixture" in r for r in result["reasons"])

    def test_incompatible_does_not_imply_winner(self, tmp_path: Path) -> None:
        """When not comparable, no mean_diff and explicit conclusion."""
        from airelab.experiments.comparison import compare_families

        d_a = tmp_path / "a"
        d_b = tmp_path / "b"
        d_a.mkdir()
        d_b.mkdir()
        _write_summary(d_a, {
            "experiment_type": "linear_regression",
            "dataset_id": "d1",
            "seeds": [42],
            "aggregated_metrics": {"test_mse": {"mean": 0.1, "std": 0.01, "min": 0.09, "max": 0.11}},
        })
        _write_summary(d_b, {
            "experiment_type": "logistic_regression",
            "dataset_id": "d2",
            "seeds": [42],
            "aggregated_metrics": {"accuracy": {"mean": 0.99, "std": 0.01, "min": 0.98, "max": 1.0}},
        })
        result = compare_families(d_a, d_b)
        assert result["comparable"] is False
        assert "conclusion" in result
        assert "No performance conclusion" in result["conclusion"]
        # No mean_diff anywhere
        for k, v in result["metric_comparisons"].items():
            assert "mean_diff" not in v, f"mean_diff found in {k}"
