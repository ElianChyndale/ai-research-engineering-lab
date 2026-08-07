"""Minimal DeepONet operator learner for C1.

DeepONet (Lu et al. 2021): G(u)(y) = sum_k branch_k(u) * trunk_k(y).
Learns the operator mapping an initial condition u0 -> solution u(t_end) at
query points y. Small, CPU/GPU-friendly. Used to test whether a surrogate
encodes solver identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BranchNet(nn.Module):
    def __init__(self, n_input: int, p: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p),
        )

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        return self.net(u0)


class TrunkNet(nn.Module):
    def __init__(self, n_coord: int, p: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_coord, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (N, n_coord) -> (N, p)."""
        return self.net(y)


class DeepONet(nn.Module):
    def __init__(self, n_input: int, n_coord: int = 1, p: int = 64, hidden: int = 64):
        super().__init__()
        self.branch = BranchNet(n_input, p, hidden)
        self.trunk = TrunkNet(n_coord, p, hidden)

    def forward(self, u0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """u0: (B, n_input); y: (B, n_query, n_coord) -> (B, n_query).

        G(u)(y) = sum_k branch_k(u0) * trunk_k(y).
        """
        b = self.branch(u0)                       # (B, p)
        B, n_query, _ = y.shape
        t = self.trunk(y.reshape(-1, y.shape[-1]))  # (B*n_query, p)
        t = t.reshape(B, n_query, -1)             # (B, n_query, p)
        return torch.sum(b.unsqueeze(1) * t, dim=2)  # (B, n_query)
