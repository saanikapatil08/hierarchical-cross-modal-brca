"""
src/visualization/attention_viz.py
===================================
Attention map visualization tools for HCMT interpretability.

Functions:
    visualize_cross_modal_attention  — Heatmap of modality-to-modality attention
    visualize_gate_weights           — Pie/bar chart of per-sample modality weights
    visualize_wsi_attention          — Overlay attention scores on WSI patch grid
    plot_training_history            — Loss and metric curves
    plot_confusion_matrix            — Styled confusion matrix heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PAM50_CLASSES = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
MODALITY_NAMES = ['WSI', 'Genomics', 'Radiology', 'Clinical']
MODALITY_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444']


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MODAL ATTENTION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def visualize_cross_modal_attention(
    attn_weights: Dict[str, Optional[torch.Tensor]],
    save_path: Optional[str] = None,
    title: str = "Cross-Modal Attention Map"
) -> plt.Figure:
    """
    Visualize cross-modal attention weights as a 2D heatmap.

    For each modality pair (query → context), shows the average attention
    weight. This reveals which modalities attend most strongly to each other.

    Args:
        attn_weights (dict): From model forward pass.
                             Keys: modality names (query), values: (B, T_q, T_ctx) or None.
                             T_ctx = sum of all other modalities' token counts.
        save_path    (str):  Optional path to save figure.
        title        (str):  Figure title.

    Returns:
        matplotlib Figure.
    """
    modality_order = ['wsi', 'genomics', 'radiology', 'clinical']
    display_names  = ['WSI', 'Genomics', 'Radiology', 'Clinical']

    # Build N×N attention matrix (mean attention from query to each other modality)
    # Approximation: divide the T_ctx dimension into per-modality segments
    N = len(modality_order)
    attn_matrix = np.zeros((N, N))

    for i, query_m in enumerate(modality_order):
        w = attn_weights.get(query_m)
        if w is None:
            continue

        # w: (B, T_q, T_ctx) — T_ctx is concatenation of other modalities
        # We approximate by assuming equal token counts for each context modality
        w_mean = w.mean(dim=(0, 1)).numpy()  # (T_ctx,)
        n_ctx_mods = N - 1
        chunk = max(1, len(w_mean) // n_ctx_mods)
        ctx_mods = [m for m in modality_order if m != query_m]
        for j_idx, ctx_m in enumerate(ctx_mods):
            j = modality_order.index(ctx_m)
            seg = w_mean[j_idx * chunk: (j_idx + 1) * chunk]
            attn_matrix[i, j] = float(seg.mean()) if len(seg) > 0 else 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        attn_matrix,
        xticklabels=display_names,
        yticklabels=display_names,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        ax=ax,
        vmin=0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Mean Attention Weight'}
    )
    ax.set_xlabel("Context Modality (Key/Value)", fontsize=12)
    ax.set_ylabel("Query Modality", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GATE WEIGHT VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_gate_weights(
    gate_weights: torch.Tensor,
    patient_ids: Optional[List[str]] = None,
    true_labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize per-sample gated modality weights as a stacked bar chart.

    Shows how much each modality contributes to the prediction for each patient,
    providing sample-level interpretability.

    Args:
        gate_weights (B, 4): Soft gate weights from model.
        patient_ids  (list): Optional patient ID labels for x-axis.
        true_labels  (B,):   Optional true PAM50 labels for title annotations.
        save_path    (str):  Optional save path.

    Returns:
        matplotlib Figure.
    """
    weights = gate_weights.numpy()  # (B, 4)
    B = weights.shape[0]

    x_labels = patient_ids if patient_ids else [f"P{i}" for i in range(B)]
    # Truncate long IDs
    x_labels = [x[:10] for x in x_labels]

    fig, ax = plt.subplots(figsize=(max(8, B * 0.6), 5))

    bottom = np.zeros(B)
    for i, (mod, color) in enumerate(zip(MODALITY_NAMES, MODALITY_COLORS)):
        ax.bar(x_labels, weights[:, i], bottom=bottom, color=color, label=mod, alpha=0.85)
        bottom += weights[:, i]

    # Annotate with true label if provided
    if true_labels is not None:
        for j, (lbl, xpos) in enumerate(zip(true_labels.tolist(), range(B))):
            ax.text(xpos, 1.02, PAM50_CLASSES[lbl], ha='center', va='bottom',
                    fontsize=8, color='black', fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Modality Weight (Gate)", fontsize=12)
    ax.set_xlabel("Patient", fontsize=12)
    ax.set_title("Per-Sample Modality Gate Weights", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_average_gate_weights_by_subtype(
    gate_weights: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Show average gate weights per PAM50 subtype.

    Reveals which modalities the model finds most informative for each subtype.
    E.g., HER2-enriched subtype might get high genomics weight because HER2
    amplification is a strong genomic signal.

    Args:
        gate_weights (B, 4): Gate weights from full test set evaluation.
        labels       (B,):   Ground truth PAM50 indices.
        save_path    (str):  Optional save path.
    """
    weights = gate_weights.numpy()
    y       = labels.numpy()

    fig, axes = plt.subplots(1, len(PAM50_CLASSES), figsize=(16, 4), sharey=True)

    for ax, cls_idx, cls_name in zip(axes, range(5), PAM50_CLASSES):
        mask = y == cls_idx
        if mask.sum() == 0:
            ax.set_title(cls_name)
            continue
        mean_w = weights[mask].mean(axis=0)
        colors = [c for c in MODALITY_COLORS]
        bars = ax.bar(MODALITY_NAMES, mean_w, color=colors, alpha=0.85, edgecolor='white')
        ax.set_ylim(0, 0.6)
        ax.set_title(f"{cls_name}\n(n={mask.sum()})", fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, mean_w):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    axes[0].set_ylabel("Mean Gate Weight", fontsize=12)
    fig.suptitle("Average Modality Contribution by PAM50 Subtype", fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list,
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot styled confusion matrix heatmap.

    Args:
        cm        (list of lists): 5×5 confusion matrix.
        normalize (bool): Show percentages (row-normalized) vs raw counts.
        save_path (str): Optional save path.

    Returns:
        matplotlib Figure.
    """
    cm_arr = np.array(cm)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_plot  = cm_arr.astype(float) / (row_sums + 1e-9)
        fmt = '.2%'
        cbar_label = 'Recall (Row-Normalized)'
    else:
        cm_plot = cm_arr
        fmt = 'd'
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_plot,
        xticklabels=PAM50_CLASSES,
        yticklabels=PAM50_CLASSES,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': cbar_label}
    )
    ax.set_xlabel("Predicted Subtype", fontsize=12)
    ax.set_ylabel("True Subtype", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HISTORY CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss + macro F1 curves.

    Args:
        history  (dict): {'train': [epoch_metrics, ...], 'val': [...]}
        save_path(str):  Optional save path.
    """
    train_loss = [e['loss'] for e in history['train']]
    val_loss   = [e['loss'] for e in history['val']]
    train_f1   = [e['macro_f1'] for e in history['train']]
    val_f1     = [e['macro_f1'] for e in history['val']]
    epochs     = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, label='Train', color='#6366f1', linewidth=2)
    ax1.plot(epochs, val_loss,   label='Val',   color='#f59e0b', linewidth=2, linestyle='--')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss", fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Macro F1
    ax2.plot(epochs, train_f1, label='Train', color='#6366f1', linewidth=2)
    ax2.plot(epochs, val_f1,   label='Val',   color='#10b981', linewidth=2, linestyle='--')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1 Score", fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
