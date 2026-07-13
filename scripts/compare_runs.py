"""CLI entry point for comparing two experiment runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from airelab.experiments.comparison import compare_runs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument("run_a", help="Path to first run directory")
    parser.add_argument("run_b", help="Path to second run directory")
    args = parser.parse_args()

    result = compare_runs(Path(args.run_a), Path(args.run_b))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
