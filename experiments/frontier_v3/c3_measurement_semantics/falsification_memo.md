# C3 Falsification Memo — MEASUREMENT-SEMANTIC FAILURE

## Result: NOT SUPPORTED in text reasoning (D), multimodal ACCESS-BLOCKED (E)

**What was tested:**
- Synthetic scientific figures (generated from known equations) with typed
  measurement ground truth (quantity, unit, axis, scale, legend, slope).
- 8 transformation families (unit mismatch, axis swap, legend swap, scale
  mismatch) + matched surface controls (synonym, rephrase).
- **Multimodal image eval: BLOCKED** — the available API does not deliver images
  ("I can't see the image because it wasn't loaded successfully"). So the
  *primary* claim (about *multimodal* models) could NOT be tested.
- **Text-only equivalent (NC2): RUN** — the model reads a figure *description*
  and judges internal consistency.

**Text-only result (honest, parse-failure-excluded):**
| Metric | Value |
|---|---|
| Original accuracy | 1.00 |
| Semantic-perturb accuracy | 0.86 |
| Matched-surface accuracy | 1.00 |
| Delta_struct | 0.14 |
| Delta_surface | 0.00 |
| **Delta_mechanism** | **0.14 (14pp)** — below the pre-registered 15pp gate |

The model **detected 12/14 (86%) of measurement-semantic perturbations**. It
does NOT "rely on surface semantic agreement while failing typed measurement
structure" — in text, it largely catches unit/axis/legend/scale inconsistencies.

**Artifact check:** the naive computation gave Delta_mechanism = 25pp (verdict A),
but this was **inflated by one parse-failure on an original** (max_tokens too
small -> empty response -> counted as wrong). Excluding parse failures, the
effect is 14pp — below the gate. This is exactly why pre-registered thresholds
+ honest parse handling matter.

**Constraint-checker intervention:** the typed measurement constraint checker
flags 16/16 semantic perturbations deterministically. But since the model
already catches 86% without it, the intervention adds little — the "missing
typed representation" is not the bottleneck in text.

**Verification of transform validity (T10/T11):** each transformation changes
exactly the intended scientific relation (unit mismatch: km/s for m/s without
conversion; axis swap: labels exchanged; legend: wrong quantity; scale: log
claimed on linear data). Surface controls preserve scientific truth (synonym/
rephrase). Confirmed by the constraint checker.

## Verdict

**C3: D — failure does not reproduce (in text reasoning).**
**Multimodal component: E — access blocked (cannot evaluate a real multimodal
model in this environment).**

The claimed *mechanism* (models encode semantic similarity better than typed
measurement structure) is **not supported** by the only testable evidence. A
frontier model in text-reasoning mode catches 86% of the semantic perturbations.

**What would make this scientifically interesting?** A real *multimodal* model
(seeing the actual figure) failing measurement-semantic perturbations much more
than surface controls, where a text-only or CoT baseline does not. That remains
untested because image input is unavailable here. But the burden of proof is on
any future multimodal claim, and the pre-registered 15pp gate is a clear bar.

## Honest limits
- Text-only is NC2 (a negative control), not the primary test. It is evidence
  against the mechanism in reasoning, but cannot rule out a *vision-specific*
  failure (e.g. a multimodal model that reads axes wrong).
- Only 1 domain of figures was used in the eval run (kinematics as the 
  constraint-checker example); the per-example data used 4 domains.
