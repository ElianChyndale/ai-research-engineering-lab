"""Tests for airelab.core.aggregation."""

from __future__ import annotations

import pytest

from airelab.core.aggregation import aggregate_floats, aggregate_metrics


@pytest.mark.unit
class TestAggregateFloats:
    def test_empty_returns_empty(self) -> None:
        assert aggregate_floats([]) == {}

    def test_single_element(self) -> None:
        result = aggregate_floats([5.0])
        assert result["mean"] == 5.0
        assert result["std"] == 0.0
        assert result["median"] == 5.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0

    def test_two_elements(self) -> None:
        result = aggregate_floats([2.0, 4.0])
        assert result["mean"] == 3.0
        assert result["min"] == 2.0
        assert result["max"] == 4.0
        assert result["median"] == 3.0
        assert result["std"] > 0

    def test_known_values(self) -> None:
        result = aggregate_floats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_all_same_values(self) -> None:
        result = aggregate_floats([7.0, 7.0, 7.0])
        assert result["mean"] == 7.0
        assert result["std"] == 0.0
        assert result["min"] == 7.0
        assert result["max"] == 7.0


@pytest.mark.unit
class TestAggregateMetrics:
    def test_empty_returns_empty(self) -> None:
        assert aggregate_metrics([]) == {}

    def test_single_metrics_dict(self) -> None:
        metrics = [{"mse": 0.5, "r2": 0.9, "name": "test"}]
        result = aggregate_metrics(metrics)
        assert result["mse"]["mean"] == 0.5
        assert result["r2"]["mean"] == 0.9
        assert result["name"] == "test"

    def test_numeric_keys_aggregated(self) -> None:
        metrics_list = [
            {"mse": 0.5, "r2": 0.9},
            {"mse": 0.6, "r2": 0.8},
            {"mse": 0.4, "r2": 0.95},
        ]
        result = aggregate_metrics(metrics_list)
        assert "mse" in result
        assert "r2" in result
        assert result["mse"]["mean"] == pytest.approx(0.5)
        assert result["r2"]["min"] == pytest.approx(0.8)

    def test_non_numeric_keys_take_first(self) -> None:
        metrics_list = [
            {"name": "run_a", "type": "lr"},
            {"name": "run_b", "type": "lr"},
        ]
        result = aggregate_metrics(metrics_list)
        assert result["name"] == "run_a"
        assert result["type"] == "lr"

    def test_mixed_numeric_and_non_numeric(self) -> None:
        metrics_list = [
            {"mse": 0.5, "name": "a"},
            {"mse": 0.6, "name": "b"},
        ]
        result = aggregate_metrics(metrics_list)
        assert result["mse"]["mean"] == pytest.approx(0.55)
        assert result["name"] == "a"

    def test_missing_key_across_dicts(self) -> None:
        metrics_list = [
            {"mse": 0.5, "extra": 1.0},
            {"mse": 0.6},
        ]
        result = aggregate_metrics(metrics_list)
        # extra only in one dict — aggregate over that one value
        assert result["extra"]["mean"] == 1.0
