"""Reproduction run stub — fill in per experiment.

The pattern (from the falsification programme):
  1. load frozen config
  2. capture provenance (git head, config hash, deps)
  3. verify seed disjointness
  4. run each method x seed via a common interface
  5. write structured records + artifact manifest
  6. evaluate pre-registered kill gates
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airelab.core.lifecycle import FrozenConfig  # noqa: E402
from airelab.core.provenance import capture_provenance, write_provenance  # noqa: E402
from airelab.cheap_kill.schemas import ExperimentSpec, ResearchHypothesis  # noqa: E402


def main() -> None:
    # 1. Hypothesis + spec (fill from paper_claim.md).
    hyp = ResearchHypothesis(
        id="TEMPLATE",
        scientific_object="",
        claim="",
        direct_owner="",
        ipw_moment="",
        strongest_baseline="",
        falsifiable_prediction="",
        negative_control="",
        primary_metric="",
        practical_effect_margin=0.25,
        pilot_budget="",
        kill_condition="",
        survival_condition="",
        confirmatory_condition="",
    )
    spec = ExperimentSpec(
        hypothesis_id=hyp.id, environment="", method="", baselines=[],
        metrics=[hyp.primary_metric],
    )
    assert spec.validate_seed_disjointness(), "seed ranges overlap"

    # 2. Provenance seal (before any seed).
    cfg = FrozenConfig.freeze(Path(__file__).parent / "config_pilot.yaml")
    assert cfg.verify(), "config changed since freeze"
    prov = capture_provenance(
        repo_root=Path(__file__).resolve().parents[1],
        config_path=cfg.path, packages=["numpy"],
    )
    write_provenance(prov, Path(__file__).parent / "provenance.json")
    print("TEMPLATE — fill in method + baselines, then run.")


if __name__ == "__main__":
    main()
