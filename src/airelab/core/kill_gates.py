"""Kill gates for cheap-falsification experiments.

An experiment specification defines PASS / WARNING / FAIL / KILL conditions
BEFORE results are analysed. Storing them up front prevents post-hoc
justification of results (the "do not tune to win" discipline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class Verdict(str):
    """A kill-gate verdict string. Purely for type clarity."""

    pass


PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"
KILL = "KILL"


@dataclass(frozen=True)
class KillGate:
    """One pre-registered gate condition.

    `condition` is a callable taking a results dict and returning a bool
    (True = gate holds / survives).
    """

    id: str
    description: str
    condition: Callable[[dict[str, Any]], bool]
    on_false: str = KILL  # what a False result means

    def evaluate(self, results: dict[str, Any]) -> str:
        return PASS if self.condition(results) else self.on_false


@dataclass
class KillGateSpec:
    """Serialisable pre-registered gate spec (saved before analysis)."""

    experiment_id: str
    gates: list[dict[str, Any]] = field(default_factory=list)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, gate_id: str, description: str, on_false: str = KILL) -> None:
        self.gates.append({"id": gate_id, "description": description, "on_false": on_false})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "KillGateSpec":
        data = json.loads(path.read_text(encoding="utf-8"))
        spec = cls(experiment_id=data["experiment_id"])
        spec.gates = data["gates"]
        spec.recorded_at = data["recorded_at"]
        return spec


def evaluate_gates(spec: KillGateSpec, conditions: dict[str, Callable[[dict[str, Any]], bool]],
                   results: dict[str, Any]) -> list[dict[str, str]]:
    """Evaluate a KillGateSpec against runtime conditions.

    `conditions` maps gate_id -> callable. Returns one row per gate.
    """
    out = []
    for gate in spec.gates:
        cond = conditions.get(gate["id"])
        if cond is None:
            out.append({"id": gate["id"], "verdict": FAIL, "reason": "no condition provided"})
            continue
        verdict = PASS if cond(results) else gate["on_false"]
        out.append({"id": gate["id"], "verdict": verdict, "reason": gate["description"]})
    return out
