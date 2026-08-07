"""Reproduction analysis stub — fill per experiment.

Pattern: aggregate per-seed records -> bootstrap CIs -> paired differences ->
evaluate pre-registered kill gates -> write KillTestReport.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airelab.core.statistics import bootstrap_ci, paired_differences  # noqa: E402
from airelab.cheap_kill.schemas import KillTestReport  # noqa: E402


def main() -> None:
    # Load per-seed records, then aggregate.
    report = KillTestReport(
        hypothesis_id="TEMPLATE", claim="", direct_owner="", counterexample="",
        baselines=[], result="", verdict="KILL", reason="not yet run",
    )
    report.save(Path(__file__).parent / "kill_test_report.json")
    print("TEMPLATE — fill in aggregation + gate evaluation.")


if __name__ == "__main__":
    main()
