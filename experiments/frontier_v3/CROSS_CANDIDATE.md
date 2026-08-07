# FRONTIER V3 — Cross-Candidate Screening Matrix

## Matrix

| | C1 cross-solver | C2 VLA action-frame | C3 measurement-semantic |
|---|---|---|---|
| **Failure reproduced?** | NO (D) | NOT RUN (E blocked) | NO in text (D); multimodal E |
| **Strong baseline survives?** | YES — DeepONet ~cross-solver invariant | n/a (blocked) | n/a (blocked for vision) |
| **Mechanism diagnostic?** | probe 0.45 < chance (no solver signal) | n/a | model catches 86% semantic (no failure) |
| **Targeted intervention works?** | not run (nothing to remove) | n/a | constraint checker flags 16/16 but model already does |
| **Negative controls pass?** | convergence test passed | n/a | parse-artifact audit revealed inflated effect |
| **General across settings?** | n/a (no failure) | n/a | 1 domain tested in eval |
| **Compute burden** | low | high (blocked) | low |
| **8-month feasibility** | n/a | blocked | low (needs real multimodal model) |
| **Theorem route** | none | none | none |
| **VERDICT** | **D — not reproduced** | **E — access blocked** | **D (text) / E (vision)** |

## Scientific priority (not total score)

1. **Mechanism survives?** — NO candidate achieved this.
2. **Strong baseline still fails?** — C1's strong baseline (cross-solver DeepONet)
   does NOT fail (it's invariant). C3's "baseline" (frontier model) catches the
   perturbations in text.
3. **Cheap reproducibility?** — C1 is cheap and reproducible (and kills itself).
4. **Theory route?** — none.

## Promotion rule

Candidate becomes PROVISIONAL RESEARCH ENGINE only on **verdict A**. **No
candidate achieved A.** Per protocol, Research Engine remains **NONE** — an
acceptable outcome.

## What is scientifically established

- **C1:** A standard operator learner (DeepONet) trained on finite-difference +
  finite-volume data generalizes to a held-out pseudo-spectral solver with a
  negligible gap (0.12× ID error) and no decodable solver identity in its
  latent. The "surrogate encodes solver artifacts" hypothesis does not
  reproduce in this controlled setting.
- **C3 (text):** A frontier model in text-reasoning mode detects 86% of
  measurement-semantic perturbations (unit/axis/legend/scale inconsistencies);
  the 15pp mechanism gate is not met (14pp, and that is a parse-artifact
  adjusted figure). The "models rely on surface semantics and fail typed
  measurement structure" hypothesis is not supported in text reasoning.
- **C2:** Blocked by access (multi-GB VLA + robot simulator required); owner
  audit shows the space is densely occupied (SE(3)-equivariant policies,
  cross-embodiment canonicalization).

## What is NOT established

- Whether a **real multimodal** model fails measurement semantics on actual
  figures (C3's primary claim) — image input unavailable here.
- Whether cross-solver encoding appears in **larger** surrogates / different
  architectures (FNO) — C1 tested one small DeepONet.

## Recommended next experiment (not auto-started)

The only unresolved scientific thread with cheap upside: **C3 multimodal** on a
real image-capable model (e.g. via a provider that delivers images), with the
15pp gate pre-registered. If image access becomes available, instantiate the
C3 multimodal eval using the existing figure generator + constraint checker.
Until then, Research Engine = NONE.
