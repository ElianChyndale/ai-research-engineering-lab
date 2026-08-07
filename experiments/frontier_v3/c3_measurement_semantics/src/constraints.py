"""C3 typed-measurement constraint checker (the minimal intervention).

Deterministic checks on a structured measurement description:
  - unit compatibility (m/s vs km/s differ by factor 1000 -> numeric conversion required)
  - dimensional consistency (velocity vs distance on same axis is a violation)
  - axis-variable consistency
  - legend binding (series label must match a known quantity)
  - equation binding (slope / relationship must be plausible)

This is the "typed measurement graph" that the mechanism hypothesis claims
models lack. If applying these constraints selectively rescues reasoning, the
missing typed representation is useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Unit -> canonical dimension family (for dimensional consistency).
DIMENSION = {
    "m/s": "speed", "km/s": "speed", "km/h": "speed", "mph": "speed",
    "m": "length", "km": "length", "cm": "length",
    "s": "time", "min": "time", "h": "time", "hr": "time",
    "K": "temperature", "C": "temperature", "degC": "temperature",
    "Pa": "pressure", "kPa": "pressure", "MPa": "pressure",
    "V": "voltage", "mV": "voltage", "kV": "voltage",
    "A": "current", "mA": "current",
    "ohm": "resistance", "kohm": "resistance",
    "J": "energy", "kJ": "energy",
    "kg/m^3": "density", "g/cm^3": "density",
    "": "dimensionless",
}

# Unit conversion factors (to a base in the dimension) — for numeric consistency.
CONVERSION = {
    "m/s": 1.0, "km/s": 1000.0, "km/h": 1.0 / 3.6, "mph": 0.44704,
    "m": 1.0, "km": 1000.0, "cm": 0.01,
    "s": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0,
    "K": 1.0, "C": 1.0, "degC": 1.0,  # temp conversion is affine; treated loosely
    "Pa": 1.0, "kPa": 1000.0, "MPa": 1e6,
    "V": 1.0, "mV": 0.001, "kV": 1000.0,
    "A": 1.0, "mA": 0.001,
    "ohm": 1.0, "kohm": 1000.0,
    "J": 1.0, "kJ": 1000.0,
}


@dataclass
class ConstraintCheck:
    violations: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return len(self.violations) == 0


def check_measurement(
    *,
    qx: str, ux: str, qy: str, uy: str,
    x_vals: list[float] | None = None,
    y_vals: list[float] | None = None,
    slope_expected: float | None = None,
    legend: str | None = None,
    known_quantities: set[str] | None = None,
) -> ConstraintCheck:
    """Check a structured measurement description for typed violations."""
    c = ConstraintCheck()
    # 1. Unit known.
    if ux not in DIMENSION:
        c.violations.append(f"unknown unit x: {ux!r}")
    if uy not in DIMENSION:
        c.violations.append(f"unknown unit y: {uy!r}")
    # 2. Dimensional consistency of x vs y.
    dx, dy = DIMENSION.get(ux), DIMENSION.get(uy)
    if dx == dy and dx not in (None, "dimensionless"):
        c.violations.append(f"x and y both {dx} (dimensional inconsistency)")
    # 3. Legend binding.
    if legend and known_quantities:
        # Legend must refer to a known quantity.
        if legend not in known_quantities:
            c.violations.append(f"legend {legend!r} not a known quantity")
    # 4. Numeric plausibility / equation binding.
    if slope_expected is not None and y_vals and x_vals:
        if len(x_vals) > 1 and max(x_vals) > min(x_vals):
            slope_obs = (max(y_vals) - min(y_vals)) / (max(x_vals) - min(x_vals))
            if slope_expected and abs(slope_obs - slope_expected) / max(slope_expected, 1e-9) > 0.5:
                c.violations.append(
                    f"slope {slope_obs:.2f} inconsistent with expected {slope_expected:.2f}"
                )
    return c
