"""Experiment type registry."""

from __future__ import annotations

from typing import Any, Callable

from airelab.core.config import ExperimentConfig

# Type alias for experiment functions
ExperimentFn = Callable[[ExperimentConfig, Any], dict[str, Any]]

_REGISTRY: dict[str, ExperimentFn] = {}


def register(name: str) -> Callable[[ExperimentFn], ExperimentFn]:
    """Decorator to register an experiment function."""
    def decorator(fn: ExperimentFn) -> ExperimentFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_experiment(name: str) -> ExperimentFn:
    """Get a registered experiment function by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown experiment type: {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_experiments() -> list[str]:
    """Return list of registered experiment names."""
    return sorted(_REGISTRY)
