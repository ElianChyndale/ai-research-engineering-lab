"""Deterministic seed control for Python random and NumPy."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Generator

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for Python random and NumPy generators."""
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def SeedContext(seed: int) -> Generator[None, None, None]:
    """Context manager that sets a seed and restores state on exit."""
    # Save state
    py_state = random.getstate()
    np_state = np.random.get_state()
    set_seed(seed)
    try:
        yield
    finally:
        # Restore state
        random.setstate(py_state)
        np.random.set_state(np_state)
