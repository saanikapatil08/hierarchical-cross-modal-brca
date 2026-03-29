"""
src/training/scheduler.py
=========================
Learning rate schedulers for HCMT training.
"""

import torch
import math
from omegaconf import DictConfig


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    """
    Build a learning rate scheduler from config.

    Supported schedulers:
        cosine_warmup — Cosine annealing with linear warmup (recommended)
        cosine        — Cosine annealing without warmup
        step          — StepLR
        plateau       — ReduceLROnPlateau

    Args:
        optimizer: The optimizer to schedule.
        cfg: Full experiment config.

    Returns:
        A PyTorch LR scheduler.
    """
    t = cfg.training
    sched_name = t.scheduler

    if sched_name == 'cosine_warmup':
        return CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=t.warmup_epochs,
            total_epochs=t.epochs,
            min_lr=t.min_lr,
            base_lr=t.lr,
        )
    elif sched_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t.epochs, eta_min=t.min_lr
        )
    elif sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif sched_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Cosine annealing LR schedule with linear warmup.

    Phase 1 (warmup): LR linearly increases from 0 → base_lr over warmup_epochs.
    Phase 2 (cosine): LR decreases from base_lr → min_lr over remaining epochs.

    This is the recommended schedule for Transformer training.

    Args:
        optimizer     : Optimizer
        warmup_epochs : Number of warmup epochs
        total_epochs  : Total training epochs
        min_lr        : Minimum LR at end of cosine decay
        base_lr       : Peak LR after warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float,
        base_lr: float,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lr       = base_lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch + 1) / float(max(1, warmup_epochs))
            else:
                # Cosine decay
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Scale to [min_lr/base_lr, 1.0]
                return min_lr / base_lr + cosine * (1.0 - min_lr / base_lr)

        super().__init__(optimizer, lr_lambda=lr_lambda)
