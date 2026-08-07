# C2 Feasibility Audit — VLA Embodiment / Action-Frame Entanglement

**Stage 0 — feasibility audit (25% budget). No huge download attempted.**

## 1. Feasible open VLA baseline

Frontier VLA baselines (RT-2, OpenVLA, π0, GR00T, RDT-1B) all require
multi-GB checkpoints and, for any real eval, a robot simulator (MuJoCo, Isaac,
or a real robot). Common requirements:

| Baseline | Checkpoint | GPU RAM (inference) | Simulator |
|---|---|---|---|
| OpenVLA 7B | ~16 GB | ~20-24 GB | BridgeData / real robot |
| RT-2-X | ~55B | not consumer | — |
| π0 | ~3B | ~16 GB | OpenVLA sim / real |
| GR00T | 1.5B-13B | ~16-24 GB | Isaac Lab |

This machine has **1 GPU (CUDA available)** but the amount and model availability
is unverified; OpenVLA/GR00T are not installed and would require multi-GB
downloads + a robot simulator not present.

## 2. Owner / collision audit

| Lineage | What it covers | Collision |
|---|---|---|
| SE(3)-equivariant visuomotor policies | eg. Equivariant Diffusion, RVT | action/camera equivariance by construction | **direct owner** |
| Cross-embodiment / action canonicalization | eg. CrossFormer, robot-independent action spaces | retargeting / canonical frames | **high** |
| Action tokenizer robustness | VLA tokenization studies | action-frame entanglement in token space | medium-high |
| Sim2real / visual robustness | domain randomization | appearance robustness (distinct from frame) | medium |

**IPW moment for C2:** a canonicalization/equivariance baseline (or action
retargeting) may already make policies frame-equivariant — if so, the claimed
"frame entanglement" is a solved engineering problem, not a research gap.

## 3. Hardware / compute verdict

- **Reproduction of a real VLA failure: NOT feasible here** without downloading
  a multi-GB checkpoint AND installing a robot simulator AND verifying GPU RAM.
  That is > reasonable screening-episode budget and would take days.
- A **lightweight equivariance probe** (does a policy satisfy
  `T_action·π(o) ≈ π(T_obs·o)`?) could in principle run on a small learned
  policy, but that would be a *toy* policy — explicitly disallowed as the sole
  baseline (rule 2: no toy-policy-only).

## 4. Verdict

**C2: E — ACCESS BLOCKED.**

A decisive reproduction requires inaccessible resources (multi-GB VLA
checkpoint + robot simulator + verified GPU RAM). Per protocol, E is NOT
scientific survival. The owner audit shows the space is densely occupied
(SE(3)-equivariant policies, cross-embodiment canonicalization) — so even if
resources were available, the "IPW moment" (canonicalization) is the likely
owner.

C2 is recorded as blocked, not killed by evidence. It would only be revisited
if a lightweight, decisive equivariance probe on a real (non-toy) policy became
feasible AND the canonicalization baseline failed to explain it.

## 5. Owner-map pointer

See `claim.md` + `preregister.yaml`. The residual scientific question (does
frame entanglement predict task failure beyond perception degradation) remains
OPEN but BLOCKED by access.
