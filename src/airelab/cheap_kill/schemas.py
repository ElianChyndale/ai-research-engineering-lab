"""Cheap-kill experiment schemas (Phase 4).

Three dataclasses a future direction fills in before coding:
  - ResearchHypothesis: the falsifiable claim + its direct owner + IPW moment.
  - ExperimentSpec: environment/method/baselines/metrics/seeds/budget.
  - KillTestReport: the filled-in result record.

The goal: once the anti-collision Top-3 returns, instantiate three cheap-kill
experiments in under a day each.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ResearchHypothesis:
    """The falsifiable claim to test cheaply."""

    id: str
    scientific_object: str  # what is the actual object of study
    claim: str  # one-sentence falsifiable claim
    direct_owner: str  # nearest existing field/paper
    ipw_moment: str  # simplest existing method/theory that could eliminate novelty
    strongest_baseline: str  # the one baseline that can kill it
    falsifiable_prediction: str  # what a cheap experiment would observe if claim true
    negative_control: str  # what would prove the mechanism is generic/trivial
    primary_metric: str  # e.g. sample_complexity_to_decision_quality
    practical_effect_margin: float  # pre-registered effect size (e.g. 0.25)
    pilot_budget: str  # seeds x configs x compute
    kill_condition: str  # the pre-registered KILL rule
    survival_condition: str  # the pre-registered survival rule
    confirmatory_condition: str  # gate before a confirmatory run

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentSpec:
    """What a cheap-kill experiment needs to run."""

    hypothesis_id: str
    environment: str  # description / class / data
    method: str  # the proposed method (minimal algorithm only)
    baselines: list[str]  # MUST baselines first
    metrics: list[str]
    seed_ranges: dict[str, list[int]] = field(default_factory=lambda: {
        "tuning": [], "pilot": [], "confirmatory": [],
    })
    compute_budget: str = ""
    artifacts: list[str] = field(default_factory=list)

    def validate_seed_disjointness(self) -> bool:
        """Programmatically reject overlapping seed ranges (Phase-2 requirement)."""
        seen: set[int] = set()
        for group, seeds in self.seed_ranges.items():
            for s in seeds:
                if s in seen:
                    return False
                seen.add(s)
        return True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KillTestReport:
    """Filled-in result of one cheap-kill experiment."""

    hypothesis_id: str
    claim: str
    direct_owner: str
    counterexample: str  # what was found
    baselines: list[str]
    result: str  # summary of the empirical result
    verdict: str  # PASS / WARNING / FAIL / KILL
    reason: str  # why this verdict

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
