"""CLI entry point for comparing two experiment families (multi-seed summaries)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from airelab.experiments.comparison import compare_families  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two experiment families")
    parser.add_argument("dir_a", help="Path to first family directory (with summary.json)")
    parser.add_argument("dir_b", help="Path to second family directory (with summary.json)")
    args = parser.parse_args()

    result = compare_families(Path(args.dir_a), Path(args.dir_b))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
