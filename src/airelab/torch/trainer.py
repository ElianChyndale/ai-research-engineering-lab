"""A small, shallow PyTorch trainer for research experiments.

Not an MLOps framework — just enough to train small neural models correctly:
train/val/test split, checkpointing, early stopping, gradient-norm logging,
optimiser/scheduler config, and reproducible reruns. A tiny synthetic dataset
is used only to verify the infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from airelab.torch.checkpoint import EarlyStopping, save_checkpoint
from airelab.torch.metrics import gradient_norm
from airelab.torch.reproducibility import set_torch_seed


@dataclass
class TrainConfig:
    """Training hyperparameters (kept shallow)."""

    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    device: str | None = None
    val_fraction: float = 0.2
    early_stop_patience: int = 5
    optimizer: str = "adam"  # adam | sgd
    weight_decay: float = 0.0
    scheduler: str | None = None  # none | step
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5


@dataclass
class TrainHistory:
    """Per-epoch training/validation metrics."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step,
                                               gamma=cfg.scheduler_gamma)
    return None


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cfg: TrainConfig,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ckpt_dir: Path | None = None,
) -> tuple[TrainHistory, torch.nn.Module]:
    """Train a model with early stopping + optional checkpointing.

    Returns (history, best_model). Uses MSE by default; pass a loss_fn for
    classification/custom losses.
    """
    set_torch_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    loss_fn = loss_fn or torch.nn.functional.mse_loss
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    es = EarlyStopping(patience=cfg.early_stop_patience, mode="min")
    history = TrainHistory()
    best_model = model  # fallback

    for epoch in range(cfg.epochs):
        model.train()
        train_loss, norms = 0.0, []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            norms.append(gradient_norm(model))
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.grad_norm.append(sum(norms) / max(len(norms), 1))

        if ckpt_dir is not None and val_loss <= (es.best if es.best is not None else float("inf")):
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer,
                            epoch=epoch, metric=val_loss)
            best_model = model

        if es.should_stop(val_loss, epoch):
            break

    return history, best_model


def split_dataset(dataset: Dataset, cfg: TrainConfig, *, seed: int | None = None) -> tuple[Dataset, Dataset]:
    """Deterministic train/validation split."""
    seed = seed or cfg.seed
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(dataset) * cfg.val_fraction)
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val], generator=g)
