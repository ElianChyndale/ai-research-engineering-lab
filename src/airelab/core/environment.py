"""Capture environment info for experiment manifests."""

from __future__ import annotations

import platform
import sys
from typing import Any


def get_environment() -> dict[str, Any]:
    """Return a snapshot of the runtime environment."""
    deps: dict[str, str] = {}
    for name in ("numpy", "pyyaml"):
        try:
            mod = __import__(name)
            deps[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[name] = "not installed"

    return {
        "python_version": sys.version,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
        "dependencies": deps,
    }
