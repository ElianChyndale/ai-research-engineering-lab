# FRONTIER V3 — Top-3 Computational Kill Sprint

Screening episode for three failure-first candidates. **This is a screening
episode, not a method-building episode.** Each candidate is tested only to the
point of a decisive kill/survive verdict.

- **C1** cross-solver physical-law disentanglement (40% budget) — executable locally.
- **C2** VLA embodiment / action-frame entanglement (25% budget) — feasibility audit only.
- **C3** scientific measurement-semantic failure (35% budget) — low compute.

## Rules

1. Failure first — reproduce the failure against strong baselines before any method.
2. Strongest baseline first — not weak/toy baselines.
3. Kill is a successful result.
4. No rescue loops — one mechanism-aligned intervention max, then KILL.
5. Verdict per candidate: A (failure+mechanism survives) / B (mixed) / C (baseline
   subsumes) / D (not reproduced) / E (blocked).

## Shared harness

Reuses `airelab` (statistics, provenance, lifecycle, kill_gates, leakage,
cheap_kill schemas, torch core). Each candidate directory has its own
`claim.md`, `owner_map.md`, `preregister.yaml`, `README.md`.
