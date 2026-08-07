"""C3: synthetic scientific figures with typed measurement ground truth.

Generates controlled scientific plots from known equations so we can apply
measurement-semantic perturbations with KNOWN correct interpretations.

Ground-truth schema per example:
  example_id, domain, quantity_x/y, unit_x/y, axis_scale, legend_mapping,
  uncertainty_definition, conditions, equation_binding, correct_answer,
  transformation_type.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOMAINS = ["kinematics", "thermodynamics", "materials", "electrical"]

# Quantity x unit pairs per domain (for typed semantics).
QUANTITY_UNITS = {
    "kinematics": [("velocity", "m/s"), ("time", "s"), ("distance", "m")],
    "thermodynamics": [("temperature", "K"), ("pressure", "Pa"), ("heat", "J")],
    "materials": [("stress", "Pa"), ("strain", ""), ("density", "kg/m^3")],
    "electrical": [("voltage", "V"), ("current", "A"), ("resistance", "ohm")],
}


def _gen_curve(domain: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a physically meaningful x,y curve with typed metadata."""
    rng = np.random.default_rng(seed)
    qx, ux = QUANTITY_UNITS[domain][0]
    qy, uy = QUANTITY_UNITS[domain][1]
    x = np.linspace(0, 10, 50)
    if qy == "pressure" and qx == "temperature":
        y = 100 + 5 * x  # ideal-gas-ish linear
    elif qy == "current" and qx == "voltage":
        y = 2 * x  # Ohm's law
    elif qy == "stress" and qx == "strain":
        y = 50 * x  # linear elastic
    elif qy == "velocity" and qx == "time":
        y = 3 * x  # constant acceleration
    else:
        y = 10 + 2 * x + 0.1 * rng.randn(50)
    meta = {"qx": qx, "ux": ux, "qy": qy, "uy": uy, "domain": domain, "slope": float(y[-1] / max(x[-1], 1e-9))}
    return x, y, meta


def render_figure(x, y, meta: dict, *, semantic_perturb: str | None = None,
                  surface_perturb: str | None = None, path) -> None:
    """Render a figure. Applies a semantic OR surface perturbation (documented)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    # Default: correct units.
    ux, uy = meta["ux"], meta["uy"]
    label = f"{meta['qy']} ({uy}) vs {meta['qx']} ({ux})"

    if semantic_perturb == "unit_mismatch":
        # Claim wrong unit without numeric conversion (e.g. km/s instead of m/s).
        uy = "km/s" if meta["uy"] == "m/s" else ("mV" if meta["uy"] == "V" else meta["uy"] + "_wrong")
    elif semantic_perturb == "axis_invert":
        # Invert axis semantics: label swapped.
        label = f"{meta['qx']} ({ux}) vs {meta['qy']} ({uy})"
    elif semantic_perturb == "legend_swap":
        label = "wrong series"  # a mislabeled series
    elif semantic_perturb == "log_axis":
        pass  # axis scale change is visual
    # Surface perturbations: cosmetically different, scientifically identical.
    if surface_perturb == "font":
        plt.rcParams["font.size"] = 14
    elif surface_perturb == "line_style":
        pass  # handled below
    elif surface_perturb == "color":
        pass

    ls = "--" if surface_perturb == "line_style" else "-"
    color = "green" if surface_perturb == "color" else "blue"
    ax.plot(x, y, ls=ls, color=color, label=label)
    ax.set_xlabel(f"{meta['qx']} ({ux})")
    ax.set_ylabel(f"{meta['qy']} ({uy})")
    if semantic_perturb == "log_axis":
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def build_dataset(n: int, out_dir, *, include_semantic: list[str], include_surface: list[str]) -> list[dict]:
    """Build n examples, each with original + semantic + surface variants."""
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        domain = DOMAINS[i % len(DOMAINS)]
        x, y, meta = _gen_curve(domain, seed=i)
        base = {"example_id": f"ex{i:03d}", **meta}
        # Original.
        orig_path = out_dir / f"{base['example_id']}_orig.png"
        render_figure(x, y, meta, path=orig_path)
        rec = {**base, "image": str(orig_path), "correct_answer": "consistent",
               "transformation": "none", "variant": "original"}
        records.append(rec)
        # Semantic variants.
        for s in include_semantic:
            p = out_dir / f"{base['example_id']}_{s}.png"
            render_figure(x, y, meta, semantic_perturb=s, path=p)
            records.append({**base, "image": str(p), "transformation": s,
                            "variant": "semantic", "correct_answer": "inconsistent" if s != "log_axis" else "consistent"})
        # Surface variants.
        for s in include_surface:
            p = out_dir / f"{base['example_id']}_{s}.png"
            render_figure(x, y, meta, surface_perturb=s, path=p)
            records.append({**base, "image": str(p), "transformation": s,
                            "variant": "surface", "correct_answer": "consistent"})
    (out_dir / "dataset.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records
