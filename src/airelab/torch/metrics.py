"""PyTorch training metrics: gradient norm logging, calibration, MSE."""

from __future__ import annotations

import torch


def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Total gradient norm across parameters with requires_grad=True."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected calibration error (ECE) for binary classification.

    probs: (N,) predicted positive-class probabilities in [0,1].
    labels: (N,) 0/1.
    """
    if probs.ndim != 1 or labels.ndim != 1:
        raise ValueError("probs and labels must be 1-D")
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must match")
    probs = probs.clamp(0.0, 1.0).detach()
    labels = labels.float().detach()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total = probs.numel()
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            continue
        bin_probs = probs[in_bin]
        bin_labels = labels[in_bin]
        conf = float(bin_probs.mean())
        acc = float(bin_labels.mean())
        ece += (n_in / total) * abs(conf - acc)
    return ece
