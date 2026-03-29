"""
src/models/fusion.py
====================
Gated modality fusion module for HCMT.

After hierarchical cross-modal attention, each modality's tokens carry enriched
representations that incorporate cross-modality context. This module aggregates
those tokens into a single fixed-size vector for classification.

Modules:
    GatedModalityFusion — Learns per-sample soft modality weights for final fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GatedModalityFusion(nn.Module):
    """
    Gated Modality Fusion with per-sample adaptive weighting.

    Motivation:
        Not all modalities are equally informative for every patient.
        - For HER2-enriched subtype: IHC status (clinical) + genomics are most relevant
        - For Triple-Negative: WSI morphology and transcriptomics are more discriminative
        - Missing or noisy modalities should be downweighted automatically

    Mechanism:
        1. Pool each modality's token sequence → scalar summary vector (B, d_model)
        2. Concatenate all summary vectors → (B, n_mod * d_model)
        3. Gate network predicts soft weights → (B, n_mod)  [sum to 1 via softmax]
        4. Weighted sum of summary vectors → fused representation (B, d_model)

    Why soft (not hard) gating?
        Soft gates are differentiable — the model learns both the gate weights
        and the representations jointly. Hard gates would require reinforcement
        learning or manual rules.

    Args:
        d_model     (int): Token dimensionality.
        n_modalities(int): Number of modalities.
        pool_type   (str): How to pool token sequences — 'mean', 'max', or 'attention'
        dropout     (float): Dropout applied before gating.

    Input:  dict of (B, T_m, d_model) tensors, one per modality
    Output:
        fused       (B, d_model) — final fused representation
        gate_weights(B, n_modalities) — soft weights (sums to 1, interpretable)
    """

    MODALITY_NAMES = ['wsi', 'genomics', 'radiology', 'clinical']

    def __init__(
        self,
        d_model: int = 256,
        n_modalities: int = 4,
        pool_type: str = 'attention',
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.pool_type = pool_type

        # Learnable linear projection per modality before pooling
        self.modal_proj = nn.ModuleDict({
            m: nn.Linear(d_model, d_model)
            for m in self.MODALITY_NAMES
        })

        # Attention pooling: one-dimensional attention over token sequence
        # Assigns importance to each token before pooling (better than mean)
        if pool_type == 'attention':
            self.attn_pool = nn.ModuleDict({
                m: nn.Sequential(
                    nn.Linear(d_model, 1),  # Scalar importance per token
                )
                for m in self.MODALITY_NAMES
            })

        # Gate network: concatenated summaries → n_modalities soft weights
        self.gate_norm = nn.LayerNorm(d_model * n_modalities)
        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * n_modalities, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_modalities)
            # Softmax applied in forward() after masking unavailable modalities
        )

        self.out_norm = nn.LayerNorm(d_model)

    def _pool_tokens(self, tokens: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Pool a token sequence (B, T, d_model) → (B, d_model).

        Three strategies:
            'mean'      — average pooling (fast, simple)
            'max'       — max pooling (captures peak signals)
            'attention' — learned attention weights over tokens (most expressive)
        """
        if self.pool_type == 'mean':
            return tokens.mean(dim=1)

        elif self.pool_type == 'max':
            return tokens.max(dim=1).values

        elif self.pool_type == 'attention':
            # Compute per-token importance scores → softmax → weighted sum
            scores = self.attn_pool[modality](tokens)   # (B, T, 1)
            weights = F.softmax(scores, dim=1)           # (B, T, 1)
            pooled = (tokens * weights).sum(dim=1)       # (B, d_model)
            return pooled

        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modalities (dict):
                Keys: modality name strings
                Values: (B, T_m, d_model) tensors, or None if absent

        Returns:
            Tuple:
                fused        (B, d_model)       — fused representation for classifier
                gate_weights (B, n_modalities)  — per-modality contribution weights
        """
        B = next(v.size(0) for v in modalities.values() if v is not None)
        device = next(v.device for v in modalities.values() if v is not None)

        # Pool each modality → (B, d_model) summary vectors
        pooled = {}
        availability_mask = torch.zeros(B, self.n_modalities, device=device)

        for i, m in enumerate(self.MODALITY_NAMES):
            if modalities.get(m) is not None:
                p = self._pool_tokens(modalities[m], m)
                pooled[m] = self.modal_proj[m](p)          # (B, d_model)
                availability_mask[:, i] = 1.0
            else:
                pooled[m] = torch.zeros(B, self.d_model, device=device)

        # Concatenate all summaries for gate input
        concat = torch.cat(
            [pooled[m] for m in self.MODALITY_NAMES],
            dim=-1
        )                                                  # (B, n_mod * d_model)

        # Gate: predict soft weights
        concat = self.gate_norm(concat)
        raw_logits = self.gate(concat)                     # (B, n_modalities)

        # Mask unavailable modalities (set their logit to -inf before softmax)
        raw_logits = raw_logits.masked_fill(
            availability_mask == 0,
            float('-inf')
        )
        gate_weights = F.softmax(raw_logits, dim=-1)       # (B, n_modalities)
        # Replace NaN with 0 (all-missing edge case)
        gate_weights = torch.nan_to_num(gate_weights, nan=0.0)

        # Weighted sum of summary vectors
        # Stack pooled: (B, n_mod, d_model)
        stacked = torch.stack([pooled[m] for m in self.MODALITY_NAMES], dim=1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        fused = self.out_norm(fused)

        return fused, gate_weights
