"""CLI entry point for validating experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from airelab.core.validation import validate_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment run artifacts")
    parser.add_argument("--run-dir", required=True, help="Path to experiment run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    result = validate_run(run_dir)
    print(result)
    if not result.valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
