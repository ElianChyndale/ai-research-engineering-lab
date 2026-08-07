# C1 Owner Map — who already owns this problem

| Lineage | Owner | What it already covers | Collision |
|---|---|---|---|
| Discretization-invariant operator learning | Li et al. FNO (2021); Kovachki et al. (2023) | learning operators across resolutions | **direct owner** |
| Domain adaptation / OOD for neural operators | DeepONet (Lu et al. 2021); various | transfer across PDE regimes | medium |
| Numerical artifact / benchmark contamination | "AI for science reproducibility" critiques | models shortcutting sim artifacts | high |
| Causality / shortcut learning | Shortcut learning lit (Geirhos et al. 2020) | models exploiting spurious features | high (conceptual) |

**IPW moment for C1:** FNO / DeepONet trained on **multiple solvers with matched
physical cases** may already be cross-solver invariant — if so, the claimed
"solver encoding" is a training-data artifact that a discretization-aware
baseline eliminates. That is the first thing to test.

**Strongest baseline:** FNO (or a DeepONet) trained on the SAME matched
cross-solver data, evaluated on the OOD solver.
