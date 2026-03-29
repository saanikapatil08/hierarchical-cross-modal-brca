"""
src/training/losses.py
======================
Loss functions for HCMT subtype classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SubtypeClassificationLoss(nn.Module):
    """
    Weighted cross-entropy loss for PAM50 subtype classification.

    PAM50 class distribution in TCGA-BRCA is heavily skewed:
        LumA ≈ 42%, LumB ≈ 28%, Basal ≈ 15%, Her2 ≈ 10%, Normal ≈ 5%

    Class weighting corrects this imbalance by assigning higher loss
    to under-represented classes.

    Label smoothing further prevents overconfidence.

    Args:
        class_weights    (Tensor, optional): (n_classes,) inverse frequency weights.
        label_smoothing  (float): Smoothing factor (0 = none, 0.1 = 10% smoothing).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  (B, n_classes): Raw model outputs (before softmax).
            targets (B,):           Ground-truth class indices.
        Returns:
            Scalar loss.
        """
        return self.ce(logits, targets)
