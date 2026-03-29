"""
scripts/train.py
================
Main training entry point for HCMT.

Supports:
    - Single run training
    - K-fold cross-validation
    - WandB experiment tracking
    - Automatic config from YAML

Usage:
    # Standard training
    python scripts/train.py --config configs/default.yaml

    # K-fold cross-validation
    python scripts/train.py --config configs/default.yaml --cv

    # Ablation: no cross-modal attention
    python scripts/train.py --config configs/default.yaml \
        ablation.disable_cross_modal_attn=true experiment.name=ablation_no_cross

    # Single modality baseline
    python scripts/train.py --config configs/default.yaml \
        data.use_wsi=true data.use_genomics=false \
        data.use_radiology=false data.use_clinical=false \
        experiment.name=wsi_only_baseline
"""

import sys
import os
import argparse
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omegaconf import OmegaConf
from pathlib import Path

from src.models.hcmt import HCMTClassifier
from src.data.dataset import build_dataloaders
from src.training.trainer import HCMTTrainer
from src.evaluation.metrics import format_metrics_table
from src.utils.utils import set_seed, load_config, get_logger

logger = get_logger(__name__)


def train_single(cfg, device, fold=None):
    """Run a single training pass (one split or one fold)."""

    fold_str = f" (fold {fold})" if fold is not None else ""
    logger.info(f"Starting training{fold_str}: {cfg.experiment.name}")

    # Build dataloaders
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg, fold=fold)

    # Build model
    logger.info("Building model...")
    model = HCMTClassifier.from_config(cfg)

    # Print model summary
    param_counts = model.get_param_count()
    logger.info("Model parameter counts:")
    for name, count in param_counts.items():
        logger.info(f"  {name:<25} {count:>10,}")

    # Build trainer
    trainer = HCMTTrainer(model, cfg, device)

    # Train
    train_results = trainer.fit(train_loader, val_loader, class_weights)

    # Load best checkpoint for test evaluation
    from src.utils.utils import load_checkpoint
    load_checkpoint(model, train_results['best_ckpt'], device=str(device))

    # Test evaluation
    logger.info("Running test set evaluation...")
    test_metrics = trainer.test(test_loader)

    # Print results
    print(format_metrics_table(test_metrics, title=f"Test Results{fold_str}"))

    return train_results, test_metrics


def train_cv(cfg, device):
    """Run k-fold cross-validation."""
    n_folds = cfg.data.n_folds
    logger.info(f"Starting {n_folds}-fold cross-validation: {cfg.experiment.name}")

    all_test_metrics = []

    for fold in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold + 1} / {n_folds}")
        logger.info(f"{'='*60}")

        # Update experiment name per fold
        fold_cfg = OmegaConf.merge(cfg, OmegaConf.create({
            'experiment': {'name': f"{cfg.experiment.name}_fold{fold}"}
        }))

        _, test_metrics = train_single(fold_cfg, device, fold=fold)
        all_test_metrics.append(test_metrics)

    # Aggregate CV results
    import numpy as np
    keys_to_avg = ['macro_f1', 'balanced_accuracy', 'mean_auc', 'cohens_kappa', 'accuracy']

    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-VALIDATION SUMMARY ({n_folds} folds)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<30} {'Mean':>10} {'Std':>10}")
    logger.info("-" * 52)
    for k in keys_to_avg:
        vals = [m[k] for m in all_test_metrics if k in m]
        if vals:
            logger.info(f"  {k:<28} {np.mean(vals):>10.4f} ± {np.std(vals):>7.4f}")

    return all_test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train HCMT for Breast Cancer Subtype Classification")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML')
    parser.add_argument('--cv', action='store_true',
                        help='Run k-fold cross-validation instead of single split')
    # Allow OmegaConf-style overrides: key=value
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides, e.g. model.d_model=512 training.lr=1e-3')
    args = parser.parse_args()

    # Load config with overrides
    cfg = load_config(args.config)
    if args.overrides:
        override_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Print final config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.experiment.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directories
    Path(cfg.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    try:
        import wandb
        if cfg.experiment.use_wandb:
            wandb.init(
                project=cfg.experiment.wandb_project,
                name=cfg.experiment.name,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
    except Exception:
        logger.info("WandB not available or disabled, skipping.")

    # Run training
    if args.cv:
        train_cv(cfg, device)
    else:
        train_single(cfg, device)


if __name__ == '__main__':
    main()
