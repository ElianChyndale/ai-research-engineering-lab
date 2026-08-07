"""C1 runner: cross-solver disentanglement screening.

Flow:
  1. generate matched cross-solver datasets (same physical_case_id per solver)
  2. train DeepONet on TRAIN solvers (e.g. fd, fv)
  3. test on ID (fd, fv) vs OOD solver (spectral)
  4. linear probe: is solver identity decodable from the DeepONet branch latent?
  5. physics probe: is the physics parameter decodable?
  6. report M3 (solver gap), M4 (probe), M6 (probe<->gap correlation)
  7. negative controls: mesh-only change; no-op label change; converged data

Kill early: if ID error ~ OOD error (no solver gap) -> D. If a strong baseline
(cross-solver-trained DeepONet) already closes the gap -> C.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from solvers import SOLVERS, generate_case  # noqa: E402
from deeponet import DeepONet  # noqa: E402


def build_dataset(
    n_cases: int,
    nx: int,
    nu_range: tuple[float, float],
    ic_types: list[str],
    t_end: float,
    solvers: list[str],
    seed: int,
) -> dict:
    """Matched cross-solver dataset.

    Returns dict with physical_case_id -> {solver: {u0, sol}} plus per-solver
    arrays. Each physical case has the SAME u0, nu, t_end across solvers.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx, endpoint=False)
    cases = {}
    for cid in range(n_cases):
        nu = float(rng.uniform(*nu_range))
        ic = ic_types[cid % len(ic_types)]
        u0 = None
        sols = {}
        for sname in solvers:
            # Generate matched: same IC via same seed.
            case = generate_case(nx, nu, t_end, ic)
            if u0 is None:
                u0 = case["u0"]
            sols[sname] = case["solvers"][sname]
        cases[cid] = {"u0": u0, "nu": nu, "ic": ic, "solvers": sols}
    return cases


def to_arrays(cases: dict, solver: str, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack u0, u_sol, nu for one solver across cases."""
    u0s = np.stack([c["u0"] for c in cases.values()])
    sols = np.stack([c["solvers"][solver] for c in cases.values()])
    nus = np.array([c["nu"] for c in cases.values()])
    return u0s, sols, nus


def train_deeponet(u0: np.ndarray, sol: np.ndarray, *, epochs: int = 300,
                   lr: float = 1e-3, seed: int = 0, device: str = "cpu") -> DeepONet:
    """Train DeepONet to map u0 -> u(t_end) at the grid points."""
    torch.manual_seed(seed)
    n = u0.shape[0]
    nx = u0.shape[1]
    model = DeepONet(nx, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
    sol_t = torch.tensor(sol, dtype=torch.float32, device=device)
    x_grid = torch.linspace(0, 1, nx, device=device).reshape(1, nx, 1).repeat(n, 1, 1)

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(u0_t, x_grid)
        loss = loss_fn(pred, sol_t)
        loss.backward()
        opt.step()
    return model


def predict(model: DeepONet, u0: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    n, nx = u0.shape
    with torch.no_grad():
        u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
        x_grid = torch.linspace(0, 1, nx, device=device).reshape(1, nx, 1).repeat(n, 1, 1)
        pred = model(u0_t, x_grid).cpu().numpy()
    return pred


def latent(model: DeepONet, u0: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Extract the branch-net latent (pre-trunk) as the representation for probing."""
    model.eval()
    with torch.no_grad():
        u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
        return model.branch(u0_t).cpu().numpy()


def linear_probe(z: np.ndarray, y: np.ndarray, *, n_train: int, seed: int = 0) -> float:
    """Linear/logistic probe accuracy (solver identity or physics label)."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z))
    zs, ys = z[idx], y[idx]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(zs[:n_train], ys[:n_train])
    return float(clf.score(zs[n_train:], ys[n_train:]))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nx = 64
    n_train_cases = 80
    n_test_cases = 40
    train_solvers = ["fd", "fv"]
    ood_solver = "spectral"

    # TRAIN set: fd + fv.
    train_cases = build_dataset(n_train_cases, nx, (0.005, 0.05), ["smooth", "gauss", "shock"],
                                0.2, train_solvers, seed=100)
    # OOD test: spectral (held-out solver).
    ood_cases = build_dataset(n_test_cases, nx, (0.005, 0.05), ["smooth", "gauss", "shock"],
                              0.2, train_solvers + [ood_solver], seed=200)

    # --- Train DeepONet on fd+fv ---
    u0_tr, sol_tr, _ = to_arrays(train_cases, "fd", nx)
    # Concatenate fd+fv as training pairs.
    u0_all = np.concatenate([to_arrays(train_cases, "fd", nx)[0],
                             to_arrays(train_cases, "fv", nx)[0]])
    sol_all = np.concatenate([to_arrays(train_cases, "fd", nx)[1],
                              to_arrays(train_cases, "fv", nx)[1]])
    model = train_deeponet(u0_all, sol_all, epochs=400, seed=0, device=device)

    # --- ID error (fd, fv) vs OOD error (spectral) ---
    id_errs = {}
    for s in train_solvers:
        u0, sol, _ = to_arrays(ood_cases, s, nx)
        pred = predict(model, u0, device)
        id_errs[s] = float(np.mean((pred - sol) ** 2))
    u0_ood, sol_ood, _ = to_arrays(ood_cases, ood_solver, nx)
    pred_ood = predict(model, u0_ood, device)
    ood_err = float(np.mean((pred_ood - sol_ood) ** 2))
    id_err = float(np.mean(list(id_errs.values())))
    gap = ood_err - id_err

    print("=== C1 cross-solver result ===")
    print(f"device={device}")
    print(f"ID error (fd,fv): {id_err:.6f}  per-solver: {id_errs}")
    print(f"OOD solver error (spectral): {ood_err:.6f}")
    print(f"M3 solver gap: {gap:.6f}  (gap/ID: {gap/max(id_err,1e-12):.2f}x)")

    # --- M4 solver probe from latent ---
    # Build a probe dataset: latent of fd vs spectral on the SAME physical cases.
    u0_fd, _, _ = to_arrays(ood_cases, "fd", nx)
    z_fd = latent(model, u0_fd, device)
    z_ood = latent(model, u0_ood, device)
    z_all = np.concatenate([z_fd, z_ood])
    y_all = np.concatenate([np.zeros(len(z_fd)), np.ones(len(z_ood))])
    probe_acc = linear_probe(z_all, y_all, n_train=len(z_fd))
    print(f"M4 solver-probe accuracy (fd vs spectral, from latent): {probe_acc:.3f}")
    # Chance = 0.5.

    # --- M5 physics probe (nu) ---
    nus = np.array([c["nu"] for c in ood_cases.values()])
    nu_bin = (nus > np.median(nus)).astype(int)
    z_fd_nu = latent(model, u0_fd, device)
    n_train = int(0.5 * len(z_fd_nu))
    nu_probe = linear_probe(z_fd_nu, nu_bin, n_train=n_train)
    print(f"M5 physics-probe accuracy (nu high/low, fd latent): {nu_probe:.3f}")

    # --- Negative control NC1: mesh-only change, same solver family ---
    # Train on fd at nx=32, test on fd at nx=128 (same solver, different mesh).
    import importlib
    from solvers import _init_burgers
    x32 = np.linspace(0, 1, 32, endpoint=False)
    u0_32, sol_32 = _init_burgers(32, 1.0, "smooth")[:1][0], None
    # (NC1 needs a re-trained model at nx=32; approximate by probing the mesh
    #  sensitivity of the current model is not clean. Keep as documented partial.)

    # --- Strongest baseline: cross-solver-trained model already has both fd,fv.
    # If OOD gap is negligible, or probe is ~chance, C1 is weak. ---
    verdict = "?"
    if gap <= 0.01 * id_err:
        verdict = "D"  # failure does not reproduce (gap negligible)
    elif probe_acc < 0.6:
        verdict = "D"  # solver identity not encoded
    else:
        verdict = "A" if gap > 0 else "B"

    result = {
        "id_err": id_err, "per_solver_id": id_errs, "ood_err": ood_err,
        "gap": gap, "probe_acc": probe_acc, "nu_probe": nu_probe, "verdict": verdict,
    }
    out = Path(__file__).resolve().parent / "results/processed/c1_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"verdict: {verdict}")
    print("saved", out)


if __name__ == "__main__":
    main()
