"""Provenance capture for research experiments.

Records git HEAD, dirty status, config hash, Python version, dependency
versions, timestamp, and hardware summary. This is the standard "seal" step
that must happen BEFORE any confirmatory seed runs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_head(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # be conservative


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_bytes().decode("utf-8", errors="replace"))


def dependency_versions(packages: list[str]) -> dict[str, str]:
    """Best-effort version of requested packages (e.g. numpy, torch)."""
    out: dict[str, str] = {}
    for name in packages:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "unavailable"
    return out


def capture_provenance(
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    packages: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture a full provenance record as a dict."""
    rec: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    if repo_root is not None:
        rec["git_head"] = git_head(repo_root)
        rec["git_dirty"] = git_dirty(repo_root)
    if config_path is not None:
        rec["config_sha256"] = sha256_file(config_path)
        rec["config_path"] = str(config_path)
    rec["dependencies"] = dependency_versions(packages or [])
    if extra:
        rec["extra"] = extra
    return rec


def write_provenance(record: dict[str, Any], out: Path) -> Path:
    """Write the provenance record as JSON (seal step)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out
