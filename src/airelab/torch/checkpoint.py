"""Model checkpointing + early stopping for small PyTorch experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, epoch: int, metric: float, extra: dict | None = None) -> Path:
    """Save model + optimizer state + metadata. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "metric": metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
    }
    torch.save(state, path)
    return path


def load_checkpoint(path: Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Load a checkpoint. Returns the saved metadata dict."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state


@dataclass
class EarlyStopping:
    """Minimal early stopping on a validation metric (lower-is-better)."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"  # 'min' or 'max'

    def __post_init__(self) -> None:
        self.best: float | None = None
        self.wait: int = 0
        self.best_epoch: int = -1

    def should_stop(self, value: float, epoch: int) -> bool:
        """Update with the latest metric; True if training should stop."""
        if self.best is None:
            self.best = value
            self.best_epoch = epoch
            return False
        better = (value < self.best - self.min_delta) if self.mode == "min" \
            else (value > self.best + self.min_delta)
        if better:
            self.best = value
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience
