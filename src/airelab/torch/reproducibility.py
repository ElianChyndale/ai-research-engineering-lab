"""Deterministic seed control + device handling for PyTorch research."""

from __future__ import annotations

import os
import random

import torch


def set_torch_seed(seed: int, *, cudnn_deterministic: bool = True) -> None:
    """Seed python, numpy, torch, and (optionally) CUDA for reproducibility.

    Use a fresh Generator per run rather than global state where possible; this
    function is for the common single-seed reproducibility case.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> torch.device:
    """Resolve the compute device: explicit, CUDA if available, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reproducible_data_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """DataLoader with a seeded Generator (no global RNG side effects)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
    )
