"""Learner-view / evaluator-view separation guard.

The critical discipline from the falsification programme: the learner must
never access evaluator-only information (hidden state, gold labels, oracle
values). This module provides a structural contract and a runtime guard so
tests can enforce it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class LeakageError(RuntimeError):
    """Raised when learner-view code accesses evaluator-only fields."""


@dataclass(frozen=True)
class ViewContract:
    """Which attribute names belong to which view."""

    learner_view: frozenset[str]
    evaluator_only: frozenset[str]

    def assert_learner_safe(self, obj: Any, *, obj_name: str = "obj") -> None:
        """Runtime guard: assert the learner object does not expose evaluator fields."""
        for attr in self.evaluator_only:
            if hasattr(obj, attr):
                raise LeakageError(
                    f"{obj_name}.{attr} is evaluator-only but accessible to the learner"
                )


@dataclass
class LearnerEvaluatorSplit:
    """Split a step result into learner view and evaluator view."""

    learner: dict[str, Any] = field(default_factory=dict)
    evaluator: dict[str, Any] = field(default_factory=dict)

    def as_learner(self) -> dict[str, Any]:
        return dict(self.learner)


def scan_learner_module_for_evaluator_fields(
    module_path: Path, evaluator_fields: set[str]
) -> list[str]:
    """AST scan: find evaluator-field accesses inside a learner module.

    Returns a list of (line, field) violations. Used by no-leakage tests to
    structurally forbid `self.state`, `obj.gold`, etc. in learner code.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in evaluator_fields:
            violations.append(f"{module_path}:{node.lineno} accesses .{node.attr}")
    return violations
