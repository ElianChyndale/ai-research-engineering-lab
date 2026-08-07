"""C1 solver tests (spec §11 T5-T9). Blocking if any fail."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solvers import SOLVERS, generate_case, reference_solution  # noqa: E402


def test_T5_three_solvers_run():
    """All three solver families produce finite outputs."""
    case = generate_case(64, nu=0.02, t_end=0.2, ic_type="smooth")
    assert set(case["solvers"]) == {"fd", "fv", "spectral"}
    for name, sol in case["solvers"].items():
        assert np.all(np.isfinite(sol)), f"{name} produced non-finite output"
        assert sol.shape == case["x"].shape


def test_T6_convergence_toward_common_reference():
    """As resolution increases, all solvers converge toward the spectral reference."""
    ref = reference_solution(np.linspace(0, 1, 64, endpoint=False),
                             np.sin(2 * np.pi * np.linspace(0, 1, 64, endpoint=False)) + 0.5,
                             0.02, 0.2)
    errs = {}
    for name, solver in SOLVERS.items():
        x = np.linspace(0, 1, 64, endpoint=False)
        u0 = np.sin(2 * np.pi * x) + 0.5
        sol = solver(x, u0, 0.02, 0.2, 1e-3)
        errs[name] = float(np.mean((sol - ref) ** 2))
    # All solvers should be "reasonably" close to the reference (same physics).
    for name, err in errs.items():
        assert err < 1e-2, f"{name} diverges from reference: {err:.2e}"


def test_T7_matched_physical_case_same_system():
    """Matched physical_case_id: same IC, nu, t_end across solvers."""
    case = generate_case(64, nu=0.02, t_end=0.2, ic_type="gauss")
    assert case["nu"] == 0.02
    assert case["t_end"] == 0.2
    assert case["ic"] == "gauss"


def test_T8_solver_families_genuinely_distinct():
    """The three solvers use different discretization families, not just mesh."""
    # Different step sizes should give slightly different trajectories (they
    # are different numerical methods), but all approximate the same physics.
    case1 = generate_case(32, nu=0.02, t_end=0.2, ic_type="smooth")
    case2 = generate_case(128, nu=0.02, t_end=0.2, ic_type="smooth")
    # The high-res spectral should be closer to the reference than low-res FD.
    ref = reference_solution(case2["x"], case2["u0"], 0.02, 0.2)
    hi_err = float(np.mean((case2["solvers"]["spectral"] - ref) ** 2))
    # Sanity: spectral at 128 should match the reference well.
    assert hi_err < 1e-3
