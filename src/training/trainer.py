"""
src/training/trainer.py
=======================
Full training loop for HCMT with:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Early stopping
    - WandB + TensorBoard logging
    - Checkpoint saving (best + latest)
    - Learning rate scheduling
    - Per-epoch metric computation
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from ..evaluation.metrics import compute_metrics
from .losses import SubtypeClassificationLoss
from .scheduler import build_scheduler
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HCMTTrainer:
    """
    Trainer for the HCMT model.

    Handles all aspects of training: forward pass, backward pass, logging,
    checkpointing, early stopping, and final evaluation.

    Args:
        model      : HCMTClassifier instance
        cfg        : OmegaConf config
        device     : torch.device

    Example:
        trainer = HCMTTrainer(model, cfg, device)
        trainer.fit(train_loader, val_loader)
        results = trainer.test(test_loader)
    """

    def __init__(self, model: nn.Module, cfg: DictConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg   = cfg
        self.device = device

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Loss — will be set before training (needs class weights from data)
        self.criterion = None

        # AMP scaler
        self.scaler = GradScaler() if cfg.training.use_amp else None

        # Metrics tracking
        self.best_metric = -float('inf')
        self.best_epoch  = 0
        self.patience_counter = 0
        self.history = {'train': [], 'val': []}

        # Logging
        self.writer = None
        if TB_AVAILABLE:
            log_dir = Path(cfg.experiment.log_dir) / cfg.experiment.name
            self.writer = SummaryWriter(str(log_dir))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with weight decay."""
        t = self.cfg.training
        # Separate params: no weight decay on norms and biases
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'norm' in name or 'bias' in name or name.endswith('.weight') and 'bn' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params,    'weight_decay': t.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=t.lr,
            betas=tuple(t.betas),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run the full training loop.

        Args:
            train_loader  : Training DataLoader
            val_loader    : Validation DataLoader
            class_weights : (n_classes,) tensor for weighted CE loss

        Returns:
            Dict with training history and best model path.
        """
        cfg = self.cfg.training

        # Set up loss
        if class_weights is not None and cfg.use_class_weights:
            class_weights = class_weights.to(self.device)
        else:
            class_weights = None

        self.criterion = SubtypeClassificationLoss(
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing
        ).to(self.device)

        # Build LR scheduler
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

        logger.info(f"Starting training for {cfg.epochs} epochs")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  AMP:    {self.cfg.training.use_amp}")

        checkpoint_dir = Path(self.cfg.experiment.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_dir / f"{self.cfg.experiment.name}_best.pth"

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader, epoch)

            # ── Validate ─────────────────────────────────────────────────────
            val_metrics = self._eval_epoch(val_loader, split='val')

            elapsed = time.time() - t0

            # ── LR Scheduler step ─────────────────────────────────────────────
            self.scheduler.step()

            # ── Logging ───────────────────────────────────────────────────────
            self._log_epoch(epoch, train_metrics, val_metrics, elapsed)

            # ── Checkpoint & Early Stopping ───────────────────────────────────
            primary_metric = val_metrics[self.cfg.evaluation.primary_metric]
            if primary_metric > self.best_metric:
                self.best_metric = primary_metric
                self.best_epoch  = epoch
                self.patience_counter = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, str(best_ckpt_path))
                logger.info(f"  ✓ New best {self.cfg.evaluation.primary_metric}: {primary_metric:.4f}")
            else:
                self.patience_counter += 1

            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            # Early stopping
            if self.patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}. Best was epoch {self.best_epoch}.")
                break

        logger.info(f"Training complete. Best epoch: {self.best_epoch}, "
                    f"Best {self.cfg.evaluation.primary_metric}: {self.best_metric:.4f}")

        if self.writer:
            self.writer.close()

        return {
            'history': self.history,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'best_ckpt': str(best_ckpt_path)
        }

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        """One training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        all_gate_weights = []

        for batch_idx, batch in enumerate(loader):
            # Move to device
            wsi       = batch['wsi'].to(self.device)       if batch['wsi']       is not None else None
            genomics  = batch['genomics'].to(self.device)  if batch['genomics']  is not None else None
            radiology = batch['radiology'].to(self.device) if batch['radiology'] is not None else None
            clinical  = batch['clinical'].to(self.device)  if batch['clinical']  is not None else None
            labels    = batch['label'].to(self.device)
            available = {k: v.to(self.device) for k, v in batch['available'].items()}

            self.optimizer.zero_grad()

            if self.cfg.training.use_amp:
                with autocast():
                    logits, gate_w, _ = self.model(wsi, genomics, radiology, clinical, available)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.clip_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, gate_w, _ = self.model(wsi, genomics, radiology, clinical, available)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.clip_grad_norm
                )
                self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(logits.detach().cpu())
            all_labels.append(labels.cpu())
            all_gate_weights.append(gate_w.detach().cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        gate_w_mean = torch.cat(all_gate_weights).mean(0).tolist()

        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(loader)
        metrics['lr']   = self.optimizer.param_groups[0]['lr']
        metrics['gate_weights'] = {
            m: w for m, w in zip(['wsi', 'genomics', 'radiology', 'clinical'], gate_w_mean)
        }
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, split: str = 'val') -> Dict:
        """One evaluation epoch (validation or test)."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            wsi       = batch['wsi'].to(self.device)       if batch['wsi']       is not None else None
            genomics  = batch['genomics'].to(self.device)  if batch['genomics']  is not None else None
            radiology = batch['radiology'].to(self.device) if batch['radiology'] is not None else None
            clinical  = batch['clinical'].to(self.device)  if batch['clinical']  is not None else None
            labels    = batch['label'].to(self.device)
            available = {k: v.to(self.device) for k, v in batch['available'].items()}

            logits, _, _ = self.model(wsi, genomics, radiology, clinical, available)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def _log_epoch(self, epoch: int, train: Dict, val: Dict, elapsed: float):
        """Print and log metrics to WandB / TensorBoard."""
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train['loss']:.4f}  F1: {train['macro_f1']:.4f} | "
            f"Val Loss: {val['loss']:.4f}  F1: {val['macro_f1']:.4f}  "
            f"BalAcc: {val['balanced_accuracy']:.4f} | "
            f"LR: {train['lr']:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard
        if self.writer:
            for k, v in train.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f'val/{k}', v, epoch)

        # WandB
        if WANDB_AVAILABLE and self.cfg.experiment.use_wandb:
            try:
                log_dict = {f'train/{k}': v for k, v in train.items() if isinstance(v, (int, float))}
                log_dict.update({f'val/{k}': v for k, v in val.items() if isinstance(v, (int, float))})
                log_dict['epoch'] = epoch
                wandb.log(log_dict)
            except Exception:
                pass

    def test(self, test_loader: DataLoader) -> Dict:
        """Run final evaluation on the test set."""
        return self._eval_epoch(test_loader, split='test')
