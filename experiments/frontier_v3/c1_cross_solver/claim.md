# C1 Claim — Cross-Solver Physical-Law Disentanglement

**One-line claim:** A scientific surrogate trained on simulated physical systems
may encode numerical-solver identity/artifacts instead of only the continuum
law, so prediction degrades when the numerical generator changes while the PDE
distribution is held fixed.

**Why this is falsifiable cheaply:** 1D Burgers/advection-diffusion, three
genuinely distinct solvers (finite difference, finite volume, pseudo-spectral),
matched counterfactual datasets (same physical_case_id across solvers), a small
operator learner, and a linear solver probe. If the unseen-solver gap is
negligible, or a discretization-aware baseline is already cross-solver
invariant, C1 is killed.

**Non-goals:** no foundation model; no 3D CFD; no novelty claim from code.
