"""Tests for airelab.core.config."""

from __future__ import annotations

import pytest

from airelab.core.config import ExperimentConfig, ExperimentType


@pytest.mark.unit
class TestExperimentConfig:
    def test_valid_linear_regression_config(self) -> None:
        cfg = ExperimentConfig(
            experiment_id="lr-001",
            experiment_type=ExperimentType.LINEAR_REGRESSION,
            seed=42,
            dataset_id="synthetic-linear",
            parameters={"learning_rate": 0.01, "epochs": 100},
            output_dir="research/results/lr-001",
        )
        assert cfg.experiment_id == "lr-001"
        assert cfg.seed == 42
        assert cfg.fixture is True  # default

    def test_missing_experiment_id_fails(self) -> None:
        with pytest.raises(ValueError, match="experiment_id"):
            ExperimentConfig(
                experiment_id="",
                experiment_type=ExperimentType.LINEAR_REGRESSION,
                seed=42,
                dataset_id="synthetic-linear",
                parameters={},
                output_dir="research/results/test",
            )

    def test_empty_dataset_id_fails(self) -> None:
        with pytest.raises(ValueError, match="dataset_id"):
            ExperimentConfig(
                experiment_id="lr-001",
                experiment_type=ExperimentType.LINEAR_REGRESSION,
                seed=42,
                dataset_id="",
                parameters={},
                output_dir="research/results/test",
            )

    def test_negative_seed_fails(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            ExperimentConfig(
                experiment_id="lr-001",
                experiment_type=ExperimentType.LINEAR_REGRESSION,
                seed=-1,
                dataset_id="synthetic-linear",
                parameters={},
                output_dir="research/results/test",
            )

    def test_non_finite_parameter_fails(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ExperimentConfig(
                experiment_id="lr-001",
                experiment_type=ExperimentType.LINEAR_REGRESSION,
                seed=42,
                dataset_id="synthetic-linear",
                parameters={"lr": float("nan")},
                output_dir="research/results/test",
            )

    def test_inf_parameter_fails(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ExperimentConfig(
                experiment_id="lr-001",
                experiment_type=ExperimentType.LINEAR_REGRESSION,
                seed=42,
                dataset_id="synthetic-linear",
                parameters={"lr": float("inf")},
                output_dir="research/results/test",
            )

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError, match="traversal"):
            ExperimentConfig(
                experiment_id="lr-001",
                experiment_type=ExperimentType.LINEAR_REGRESSION,
                seed=42,
                dataset_id="synthetic-linear",
                parameters={},
                output_dir="../outside",
            )

    def test_unknown_experiment_type_fails(self) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(
                experiment_id="lr-001",
                experiment_type="unknown_type",  # type: ignore[arg-type]
                seed=42,
                dataset_id="synthetic-linear",
                parameters={},
                output_dir="research/results/test",
            )

    def test_fixture_defaults_true(self) -> None:
        cfg = ExperimentConfig(
            experiment_id="lr-001",
            experiment_type=ExperimentType.LINEAR_REGRESSION,
            seed=42,
            dataset_id="synthetic-linear",
            parameters={},
            output_dir="research/results/test",
        )
        assert cfg.fixture is True

    def test_config_to_dict(self) -> None:
        cfg = ExperimentConfig(
            experiment_id="lr-001",
            experiment_type=ExperimentType.LINEAR_REGRESSION,
            seed=42,
            dataset_id="synthetic-linear",
            parameters={"lr": 0.01},
            output_dir="research/results/test",
        )
        d = cfg.to_dict()
        assert d["experiment_id"] == "lr-001"
        assert d["seed"] == 42
        assert d["fixture"] is True
