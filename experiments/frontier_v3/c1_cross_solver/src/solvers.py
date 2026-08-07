"""Three genuinely distinct numerical solvers for 1D viscous Burgers.

    u_t + u u_x = nu u_xx

Cross-solver means genuinely different DISCRETIZATION FAMILIES, not just
different mesh resolutions:

  S1 finite difference   — explicit upwind + central diffusion (FTCS/upwind)
  S2 finite volume       — Godunov-type with Rusanov flux + central diffusion
  S3 pseudo-spectral     — Fourier spectral in space, exponential/forward Euler in time

All three solve the SAME physical system (same PDE, coefficients, IC, BC, t),
so a matched physical_case_id pairs the same continuum solution across solvers.
A high-resolution spectral reference is used for convergence checks.
"""

from __future__ import annotations

import numpy as np


def _init_burgers(nx: int, L: float, ic_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return grid (x) and initial condition (u0) on [0, L]."""
    x = np.linspace(0.0, L, nx, endpoint=False)
    if ic_type == "smooth":
        u0 = np.sin(2 * np.pi * x / L) + 0.5
    elif ic_type == "shock":
        u0 = np.where(x < L / 2, 1.0, 0.5)
    elif ic_type == "gauss":
        u0 = 0.5 + np.exp(-((x - L / 2) ** 2) / 0.02)
    else:
        raise ValueError(ic_type)
    return x, u0


def solve_fd(x: np.ndarray, u0: np.ndarray, nu: float, t_end: float, dt: float) -> np.ndarray:
    """S1 finite difference: upwind advection + central diffusion (FTCS).

    Conservative-upwind form avoids the stability pathology of pure central
    advection at low nu.
    """
    nx = len(x)
    dx = x[1] - x[0]
    u = u0.copy()
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        u_prev = u.copy()
        # Upwind advection (u>0 assumed; flux u^2/2 via upwind).
        for i in range(1, nx):
            u[i] = u_prev[i] - (dt / dx) * (0.5 * u_prev[i] ** 2 - 0.5 * u_prev[i - 1] ** 2)
        u[0] = u_prev[0] - (dt / dx) * (0.5 * u_prev[0] ** 2 - 0.5 * u_prev[-1] ** 2)  # periodic
        # Central diffusion.
        for i in range(nx):
            u[i] = u[i] + (nu * dt / dx**2) * (u[(i + 1) % nx] - 2 * u[i] + u[(i - 1) % nx])
    return u


def solve_fv(x: np.ndarray, u0: np.ndarray, nu: float, t_end: float, dt: float) -> np.ndarray:
    """S2 finite volume: Godunov with Rusanov flux + central diffusion.

    Cell averages with periodic BCs. Different discretization family from FD:
    conservation form with numerical flux, not pointwise upwind.
    """
    nx = len(x)
    dx = x[1] - x[0]
    u = u0.copy()
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        # Rusanov flux F_ij = 0.5*(f_i+f_j) - 0.5*max(|u_i|,|u_j|)*(u_j-u_i), f=u^2/2.
        flux = np.zeros(nx)
        for i in range(nx):
            ul, ur = u[i], u[(i + 1) % nx]
            smax = max(abs(ul), abs(ur))
            flux[i] = 0.5 * (0.5 * ul**2 + 0.5 * ur**2) - 0.5 * smax * (ur - ul)
        u_new = u.copy()
        for i in range(nx):
            u_new[i] = u[i] - (dt / dx) * (flux[i] - flux[(i - 1) % nx])
        # Central diffusion.
        for i in range(nx):
            u_new[i] = u_new[i] + (nu * dt / dx**2) * (u[(i + 1) % nx] - 2 * u[i] + u[(i - 1) % nx])
        u = u_new
    return u


def solve_spectral(x: np.ndarray, u0: np.ndarray, nu: float, t_end: float, dt: float) -> np.ndarray:
    """S3 pseudo-spectral: Fourier spatial + integrating-factor diffusion + RK2.

    Different family: global spectral accuracy, no upwinding. The diffusion is
    handled exactly via the integrating factor exp(-nu k^2 t) so only the
    advection nonlinearity is stepped.
    """
    nx = len(x)
    L = x[-1] - x[0] + (x[1] - x[0])
    k = 2 * np.pi * np.fft.fftfreq(nx, d=(x[1] - x[0]))
    k[0] = 0.0
    u_hat = np.fft.fft(u0)
    n_steps = int(t_end / dt)

    def adv(u):
        # u u_x in physical space.
        return u * np.real(np.fft.ifft(1j * k * np.fft.fft(u)))

    u_phys = u0.copy()
    for n in range(n_steps):
        # RK2 with integrating factor on diffusion.
        half = np.exp(-nu * k**2 * dt / 2)
        full = half * half
        # Step 1
        a = adv(u_phys)
        u1_hat = (np.fft.fft(u_phys) * half) - dt * 0.5 * np.fft.fft(a)
        u1 = np.real(np.fft.ifft(u1_hat))
        # Step 2
        b = adv(u1)
        u_hat = np.fft.fft(u_phys) * full - dt * np.fft.fft(b) * half
        u_phys = np.real(np.fft.ifft(u_hat))
    return u_phys


def reference_solution(x: np.ndarray, u0: np.ndarray, nu: float, t_end: float) -> np.ndarray:
    """High-accuracy spectral reference (fine dt, exact diffusion)."""
    dt = min(1e-4, t_end / 2000)
    return solve_spectral(x, u0, nu, t_end, dt)


SOLVERS = {
    "fd": solve_fd,
    "fv": solve_fv,
    "spectral": solve_spectral,
}


def generate_case(nx: int, nu: float, t_end: float, ic_type: str, L: float = 1.0) -> dict:
    """Generate one physical case with all three solvers (matched physical_case_id)."""
    x, u0 = _init_burgers(nx, L, ic_type)
    dt = min(0.5 * (x[1] - x[0]) / 2.0, 1e-3)  # stable-ish for all
    sols = {name: solver(x.copy(), u0.copy(), nu, t_end, dt)
            for name, solver in SOLVERS.items()}
    return {"x": x, "u0": u0, "solvers": sols, "nu": nu, "t_end": t_end, "ic": ic_type}
