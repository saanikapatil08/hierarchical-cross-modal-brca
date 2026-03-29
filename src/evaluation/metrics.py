"""
src/evaluation/metrics.py
=========================
Evaluation metrics for PAM50 subtype classification.

Metrics:
    - Macro F1 score (primary metric — handles class imbalance)
    - Balanced accuracy
    - Per-class AUC (one-vs-rest)
    - Cohen's Kappa
    - Per-class precision, recall, F1
    - Confusion matrix
"""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize


PAM50_CLASSES = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    return_confusion_matrix: bool = True
) -> Dict:
    """
    Compute all classification metrics from logits and ground-truth labels.

    Args:
        logits  (B, n_classes): Raw model outputs (before softmax).
        labels  (B,):           Ground-truth class indices.
        return_confusion_matrix (bool): Whether to include confusion matrix.

    Returns:
        Dict with all computed metrics.
    """
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    y_true = labels.numpy()

    metrics = {}

    # ── Primary metrics ───────────────────────────────────────────────────────
    metrics['macro_f1'] = f1_score(y_true, preds, average='macro', zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, preds)

    # ── Per-class AUC (one-vs-rest) ───────────────────────────────────────────
    try:
        y_bin = label_binarize(y_true, classes=list(range(5)))
        if y_bin.shape[1] == 5:  # All classes present
            per_class_auc = roc_auc_score(y_bin, probs, average=None, multi_class='ovr')
            metrics['mean_auc'] = float(per_class_auc.mean())
            for i, cls in enumerate(PAM50_CLASSES):
                metrics[f'auc_{cls}'] = float(per_class_auc[i])
        else:
            metrics['mean_auc'] = float('nan')
    except Exception:
        metrics['mean_auc'] = float('nan')

    # ── Cohen's Kappa ─────────────────────────────────────────────────────────
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, preds)

    # ── Per-class precision, recall, F1 ──────────────────────────────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, average=None, zero_division=0, labels=list(range(5))
    )
    for i, cls in enumerate(PAM50_CLASSES):
        metrics[f'precision_{cls}'] = float(precision[i])
        metrics[f'recall_{cls}']    = float(recall[i])
        metrics[f'f1_{cls}']        = float(f1[i])
        metrics[f'support_{cls}']   = int(support[i])

    # ── Macro averages ────────────────────────────────────────────────────────
    metrics['macro_precision'] = float(precision.mean())
    metrics['macro_recall']    = float(recall.mean())

    # ── Confusion matrix ─────────────────────────────────────────────────────
    if return_confusion_matrix:
        metrics['confusion_matrix'] = confusion_matrix(y_true, preds, labels=list(range(5))).tolist()

    # ── Accuracy ──────────────────────────────────────────────────────────────
    metrics['accuracy'] = float((preds == y_true).mean())

    return metrics


def format_metrics_table(metrics: Dict, title: str = "Evaluation Results") -> str:
    """
    Format metrics dict as a readable table string.

    Returns:
        Multi-line string with formatted metrics table.
    """
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]

    # Summary metrics
    summary_keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'mean_auc', 'cohens_kappa']
    lines.append(f"\n{'Metric':<30} {'Value':>10}")
    lines.append("-" * 42)
    for k in summary_keys:
        if k in metrics:
            lines.append(f"  {k:<28} {metrics[k]:>10.4f}")

    # Per-class metrics
    lines.append(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Support':>10}")
    lines.append("-" * 62)
    for cls in PAM50_CLASSES:
        p = metrics.get(f'precision_{cls}', 0)
        r = metrics.get(f'recall_{cls}', 0)
        f = metrics.get(f'f1_{cls}', 0)
        a = metrics.get(f'auc_{cls}', float('nan'))
        s = metrics.get(f'support_{cls}', 0)
        lines.append(f"  {cls:<8} {p:>10.4f} {r:>10.4f} {f:>10.4f} {a:>10.4f} {s:>10}")

    # Confusion matrix
    if 'confusion_matrix' in metrics:
        lines.append(f"\nConfusion Matrix (rows=True, cols=Pred):")
        lines.append(f"  {'':8}" + "".join(f"{c:>8}" for c in PAM50_CLASSES))
        for i, cls in enumerate(PAM50_CLASSES):
            row = metrics['confusion_matrix'][i]
            lines.append(f"  {cls:<8}" + "".join(f"{v:>8}" for v in row))

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)
