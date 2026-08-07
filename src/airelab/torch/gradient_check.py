"""Finite-difference gradient check for a tiny differentiable module.

Verifies that autodiff gradients agree with central finite differences —
a cheap correctness gate before trusting any learned model.
"""

from __future__ import annotations

import torch


def gradient_check(
    module,
    x: torch.Tensor,
    *,
    eps: float = 1e-2,
    tol: float = 1e-2,
    atol: float = 1e-3,
) -> tuple[bool, float]:
    """Compare autodiff gradients against central finite differences.

    `module` is a callable x -> scalar loss (a model wrapped with a loss, or a
    function). Returns (passed, max_abs_error). A fresh leaf input is used so
    the check is self-contained and re-runnable.
    """
    x = x.detach().clone().requires_grad_(True)  # fresh leaf

    # Autodiff gradient.
    loss = module(x)
    if loss.numel() != 1:
        raise ValueError("gradient_check requires a scalar loss")
    loss.backward()
    if x.grad is None:
        raise RuntimeError("autodiff produced no gradient for the input")
    grad_auto = x.grad.detach().clone()

    # Central finite difference per element (no autograd graphs in the FD pass).
    grad_fd = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            xp = x.detach().clone().view(-1)
            xm = x.detach().clone().view(-1)
            xp[i] += eps
            xm[i] -= eps
            lp = module(xp.view_as(x))
            lm = module(xm.view_as(x))
            grad_fd.view(-1)[i] = (lp - lm) / (2 * eps)

    # Mixed absolute/relative tolerance (torch.autograd.gradcheck convention).
    diff = torch.abs(grad_auto - grad_fd)
    scale = torch.abs(grad_auto)
    ok = bool(torch.all(diff <= atol + tol * scale))
    return ok, diff.max().item()
