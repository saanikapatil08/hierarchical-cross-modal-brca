"""
scripts/evaluate.py
===================
Full evaluation pipeline for a trained HCMT model.

Runs on a test set and produces:
    - Classification metrics table (F1, AUC, Kappa, etc.)
    - Confusion matrix plot
    - Gate weight visualizations (per-sample + per-subtype)
    - Cross-modal attention heatmaps
    - Per-patient prediction CSV

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/hcmt_baseline_best.pth \
        --config configs/default.yaml \
        --output_dir outputs/evaluation

    # Evaluate on a specific fold's test set
    python scripts/evaluate.py \
        --checkpoint checkpoints/hcmt_baseline_fold0_best.pth \
        --config configs/default.yaml \
        --fold 0
"""

import sys
import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hcmt import HCMTClassifier, PAM50_CLASSES
from src.data.dataset import build_dataloaders
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.visualization.attention_viz import (
    plot_confusion_matrix,
    visualize_gate_weights,
    visualize_average_gate_weights_by_subtype,
    visualize_cross_modal_attention,
    plot_training_history
)
from src.utils.utils import set_seed, load_config, load_checkpoint, get_logger

logger = get_logger(__name__)


@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model inference on a DataLoader and collect all predictions.

    Returns:
        all_logits      (N, 5)   — raw logits
        all_labels      (N,)     — ground truth
        all_gate_weights(N, 4)   — modality gate weights
        all_attn        list     — attention weight dicts per batch
        all_patient_ids list     — patient ID strings
    """
    model.eval()
    all_logits, all_labels, all_gate_weights = [], [], []
    all_attn, all_patient_ids = [], []

    for batch in loader:
        wsi       = batch['wsi'].to(device)       if batch['wsi']       is not None else None
        genomics  = batch['genomics'].to(device)  if batch['genomics']  is not None else None
        radiology = batch['radiology'].to(device) if batch['radiology'] is not None else None
        clinical  = batch['clinical'].to(device)  if batch['clinical']  is not None else None
        labels    = batch['label']
        available = {k: v.to(device) for k, v in batch['available'].items()}

        logits, gate_w, attn = model(wsi, genomics, radiology, clinical, available)

        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_gate_weights.append(gate_w.cpu())
        all_attn.append(attn)
        all_patient_ids.extend(batch['patient_ids'])

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
        torch.cat(all_gate_weights),
        all_attn,
        all_patient_ids
    )


def save_predictions_csv(
    patient_ids, logits, labels, gate_weights, output_path
):
    """Save per-patient predictions and gate weights to CSV."""
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    y_true = labels.numpy()

    rows = []
    for i, pid in enumerate(patient_ids):
        row = {
            'patient_id': pid,
            'true_label': PAM50_CLASSES[y_true[i]],
            'pred_label': PAM50_CLASSES[preds[i]],
            'correct':    bool(preds[i] == y_true[i]),
        }
        for j, cls in enumerate(PAM50_CLASSES):
            row[f'prob_{cls}'] = float(probs[i, j])
        for j, mod in enumerate(['wsi', 'genomics', 'radiology', 'clinical']):
            row[f'gate_{mod}'] = float(gate_weights[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained HCMT model")
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--config',     required=True, help='Path to config YAML')
    parser.add_argument('--output_dir', default='outputs/evaluation')
    parser.add_argument('--fold',       type=int, default=None, help='Fold for CV evaluation')
    parser.add_argument('--split',      default='test', choices=['val', 'test'],
                        help='Which split to evaluate on')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = HCMTClassifier.from_config(cfg)
    ckpt_info = load_checkpoint(model, args.checkpoint, device=str(device))
    model = model.to(device)
    logger.info(f"  Checkpoint epoch: {ckpt_info['epoch']}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg, fold=args.fold)
    loader = test_loader if args.split == 'test' else val_loader

    # ── Run inference ─────────────────────────────────────────────────────────
    logger.info("Running inference...")
    logits, labels, gate_weights, attn_list, patient_ids = run_inference(model, loader, device)

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(logits, labels)
    print(format_metrics_table(metrics, title="HCMT Evaluation Results"))

    # Save metrics JSON
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    logger.info(f"Metrics saved to: {output_dir / 'metrics.json'}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    logger.info("Plotting confusion matrix...")
    fig = plot_confusion_matrix(
        metrics['confusion_matrix'],
        normalize=True,
        save_path=str(output_dir / 'confusion_matrix.png'),
        title="HCMT — PAM50 Subtype Classification"
    )
    fig.savefig(output_dir / 'confusion_matrix.pdf', bbox_inches='tight')

    # ── Gate weight visualizations ────────────────────────────────────────────
    logger.info("Plotting gate weights...")
    # Show first 20 patients
    n_show = min(20, len(patient_ids))
    fig = visualize_gate_weights(
        gate_weights[:n_show],
        patient_ids=patient_ids[:n_show],
        true_labels=labels[:n_show],
        save_path=str(output_dir / 'gate_weights_samples.png')
    )

    # Average gate weights per subtype
    fig2 = visualize_average_gate_weights_by_subtype(
        gate_weights, labels,
        save_path=str(output_dir / 'gate_weights_by_subtype.png')
    )

    # ── Cross-modal attention ──────────────────────────────────────────────────
    if attn_list and any(v is not None for v in attn_list[0].values()):
        logger.info("Plotting cross-modal attention...")
        # Average attention over first batch
        first_batch_attn = attn_list[0]
        fig3 = visualize_cross_modal_attention(
            first_batch_attn,
            save_path=str(output_dir / 'cross_modal_attention.png'),
            title="HCMT — Cross-Modal Attention Map"
        )

    # ── Save per-patient predictions ──────────────────────────────────────────
    logger.info("Saving predictions CSV...")
    save_predictions_csv(
        patient_ids, logits, labels, gate_weights,
        output_path=str(output_dir / 'predictions.csv')
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation complete. Results in: {output_dir}")
    logger.info(f"  Macro F1:           {metrics['macro_f1']:.4f}")
    logger.info(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Mean AUC:           {metrics['mean_auc']:.4f}")
    logger.info(f"  Cohen's Kappa:      {metrics['cohens_kappa']:.4f}")
    logger.info(f"{'='*50}")


if __name__ == '__main__':
    main()
