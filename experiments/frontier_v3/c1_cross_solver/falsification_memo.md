# C1 Falsification Memo — CROSS-SOLVER DISENTANGLEMENT

## Result: KILLED (D — failure does not reproduce)

**What was tested:**
- 1D viscous Burgers, 3 genuinely distinct solver families: finite difference
  (S1), finite volume/Rusanov (S2), pseudo-spectral (S3). Verified they solve
  the same physics (T6 convergence test passed).
- Trained a small DeepONet on TRAIN solvers fd+fv; tested on OOD solver
  spectral. Matched physical_case_id across solvers.

**Numbers:**
- ID error (fd, fv): 0.0369
- OOD solver error (spectral): 0.0415
- **M3 solver gap = 0.0046 = 0.12× ID** — negligible.
- **M4 solver-probe accuracy = 0.450** — BELOW chance (0.5). The DeepONet
  latent does NOT encode solver identity.
- M5 physics-probe (nu high/low) = 0.600 (weak).

**Why this is decisive:**
Two independent kill conditions both hold:
1. **Unseen-solver gap is negligible** (0.12× ID) — the surrogate generalizes
   across solver families without measurable degradation.
2. **Solver identity is not encoded** (probe 0.45 < chance) — even if there were
   a gap, there is no decodable solver signal in the representation.

The claimed mechanism (z ≈ z_physics + z_solver + z_interaction) is NOT
supported: there is no z_solver to disentangle. The hypothesis that surrogates
"encode numerical-solver artifacts instead of the continuum law" does not
reproduce in this minimal, well-controlled setting.

**Per sprint rule:** decisive D → stop this branch immediately. No intervention
built (a paired solver-consistency loss would have nothing to remove).

**Possible objections / honest limits:**
- The DeepONet is small and trained on only 160 pairs (fd+fv); a different
  architecture (FNO) or more data might behave differently. But the burden of
  proof is now on any future cross-solver claim, against the finding that a
  standard operator learner is already ~cross-solver invariant here.
- The spectral solver at this resolution/dt may be "too close" to the physics;
  NC4 (converged solutions) was intended to test this but became moot because
  even the raw OOD gap is negligible.

## What result would make this scientifically boring?
This result IS boring-for-C1: the failure does not reproduce; the mechanism is
unsupported. That is a successful kill.

## Verdict
**C1 VERDICT: D — FAILURE DOES NOT REPRODUCE.**
