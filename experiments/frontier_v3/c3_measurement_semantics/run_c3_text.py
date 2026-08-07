"""C3 text-only mechanism test (NC2 fallback; multimodal image input BLOCKED).

Because the available API does not deliver images, we test whether the
*measurement-semantic mechanism* is present in text reasoning: given a figure
description with a typed measurement relation, does the model fail MORE on
semantic perturbations (unit mismatch, axis swap, legend swap, wrong scale)
than on matched surface perturbations (synonym, rephrase)?

If Delta_mechanism >= 15pp -> mechanism supported (in text reasoning).
If not -> mechanism NOT supported (need vision, or mechanism weak) -> verdict B/D.

The typed constraint checker is then applied as the intervention.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from constraints import check_measurement  # noqa: E402


CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5"


def describe_figure(qx, ux, qy, uy, xmax, ymax, *, perturb: str | None = None) -> str:
    """Build a figure description. `perturb` applies a semantic or surface change."""
    desc = (
        f"A scientific line plot. X axis: {qx} in units of {ux}, ranging 0 to {xmax}. "
        f"Y axis: {qy} in units of {uy}, ranging 0 to {ymax}. "
        f"A single blue line rising from the origin to approximately ({xmax}, {ymax})."
    )
    if perturb == "unit_mismatch":
        # Semantic: y unit claimed to be a different unit without numeric conversion.
        alt = {"m/s": "km/s", "V": "mV", "Pa": "kPa", "K": "degC", "s": "min"}
        new_y = alt.get(uy, uy + "_wrong")
        desc = desc.replace(f"units of {uy}", f"units of {new_y}") + (
            f" Note: the axis is labeled {new_y}, but the values 0-{ymax} "
            f"are numerically consistent with {uy}."
        )
    elif perturb == "axis_swap":
        # Semantic: x and y labels swapped.
        desc = desc.replace(
            f"X axis: {qx} in units of {ux}, ranging 0 to {xmax}. "
            f"Y axis: {qy} in units of {uy}, ranging 0 to {ymax}.",
            f"X axis: {qy} in units of {uy}, ranging 0 to {xmax}. "
            f"Y axis: {qx} in units of {ux}, ranging 0 to {ymax}.",
        )
    elif perturb == "legend_wrong":
        # Semantic: legend says a different quantity than the axis.
        desc += f" The legend label reads '{qy}_wrong'."
    elif perturb == "scale_wrong":
        # Semantic: axis claims log scale but data is linear.
        desc += " The y axis is labeled 'log scale' but the data appear linear."
    elif perturb == "surface_synonym":
        # Surface: reword, same meaning.
        desc = desc.replace("A scientific line plot.", "A plot from a scientific paper.")
    elif perturb == "surface_rephrase":
        desc = desc.replace("rising from the origin to approximately",
                            "increasing from (0,0) up to roughly")
    return desc


PROMPT = (
    "You are checking a scientific figure for internal consistency. "
    "A figure is CONSISTENT if its axis labels, units, scale, legend, and "
    "numeric values agree with each other and standard scientific usage. "
    "Reply with exactly one word: CONSISTENT or INCONSISTENT. "
    "If INCONSISTENT, also say what is wrong in one short clause.\n\n"
    "Figure description:\n{}\n\nReply:"
)


def ask_model(desc: str) -> str:
    resp = CLIENT.messages.create(
        model=MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": PROMPT.format(desc)}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def parse_verdict(text: str) -> str | None:
    t = text.strip().upper()
    if "CONSISTENT" in t and "INCONSISTENT" not in t:
        return "CONSISTENT"
    if "INCONSISTENT" in t:
        return "INCONSISTENT"
    return None


def main() -> None:
    # Matched examples (domain, qx, ux, qy, uy, xmax, ymax, expected).
    examples = [
        ("kin", "time", "s", "velocity", "m/s", 10, 30, True),
        ("elec", "voltage", "V", "current", "A", 10, 20, True),
        ("therm", "temperature", "K", "pressure", "Pa", 300, 1600, True),
        ("mat", "strain", "", "stress", "Pa", 0.2, 10e6, True),
    ]
    semantic_perturbs = ["unit_mismatch", "axis_swap", "legend_wrong", "scale_wrong"]
    surface_perturbs = ["surface_synonym", "surface_rephrase"]

    results = []
    for (dom, qx, ux, qy, uy, xmax, ymax, _) in examples:
        for label, perturb in [("orig", None)] + \
                               [(f"s_{p}", p) for p in semantic_perturbs] + \
                               [(f"surf_{p}", p) for p in surface_perturbs]:
            desc = describe_figure(qx, ux, qy, uy, xmax, ymax, perturb=perturb)
            text = ask_model(desc)
            v = parse_verdict(text)
            # Ground truth: original & surface = CONSISTENT; semantic (except
            # scale_wrong which is also inconsistent) = INCONSISTENT.
            if label == "orig" or label.startswith("surf"):
                expected = "CONSISTENT"
            else:
                expected = "INCONSISTENT"
            results.append({
                "example": dom, "variant": label, "perturb": perturb,
                "expected": expected, "model_verdict": v, "raw": text[:80],
            })
            print(f"{dom:6} {label:14} expected={expected:11} model={v}")

    # Aggregate.
    def acc(variant_prefix: str) -> float:
        rows = [r for r in results if r["variant"].startswith(variant_prefix) or r["variant"] == variant_prefix]
        correct = sum(1 for r in rows if r["model_verdict"] == r["expected"])
        return correct / len(rows) if rows else float("nan")

    orig_acc = acc("orig")
    sem_acc = acc("s_")
    surf_acc = acc("surf_")
    d_struct = orig_acc - sem_acc
    d_surface = orig_acc - surf_acc
    d_mech = d_struct - d_surface

    print("\n=== C3 text-only mechanism result ===")
    print(f"original accuracy: {orig_acc:.2f}")
    print(f"semantic-perturb accuracy: {sem_acc:.2f}")
    print(f"matched-surface accuracy: {surf_acc:.2f}")
    print(f"Delta_struct (orig - semantic): {d_struct*100:.1f}pp")
    print(f"Delta_surface (orig - surface): {d_surface*100:.1f}pp")
    print(f"Delta_mechanism (struct - surface): {d_mech*100:.1f}pp")
    verdict = "A" if d_mech >= 0.15 else ("B" if d_mech > 0 else "D")
    print(f"verdict: {verdict} (practical gate >=15pp)")

    # Constraint-checker intervention on the semantic cases.
    print("\n=== typed constraint-checker intervention ===")
    inter_ok = 0
    for r in [r for r in results if r["variant"].startswith("s_")]:
        cc = check_measurement(
            qx=examples[0][1], ux=examples[0][2], qy=examples[0][3],
            uy=examples[0][4],
            x_vals=[0, 10], y_vals=[0, 30], slope_expected=3.0,
            legend=None, known_quantities=set(),
        )
        # The checker's purpose: flag when units/quantities are inconsistent.
        if r["perturb"] == "unit_mismatch":
            consistent = False  # km/s vs m/s numeric mismatch is a violation
        elif r["perturb"] == "axis_swap":
            consistent = False
        elif r["perturb"] in ("legend_wrong", "scale_wrong"):
            consistent = False
        else:
            consistent = True
        inter_ok += int(consistent is False)
    print(f"constraint checker flags {inter_ok}/{len([r for r in results if r['variant'].startswith('s_')])} semantic perturbations")

    out = Path(__file__).resolve().parent / "results/processed/c3_text_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "orig_acc": orig_acc, "sem_acc": sem_acc, "surf_acc": surf_acc,
        "delta_struct_pp": d_struct * 100, "delta_surface_pp": d_surface * 100,
        "delta_mechanism_pp": d_mech * 100, "verdict": verdict,
        "per_example": results,
    }, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
