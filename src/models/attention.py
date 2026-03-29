"""
src/models/attention.py
=======================
Attention modules for HCMT.

Modules:
    IntraModalTransformer      — Standard Transformer encoder within one modality
    CrossModalAttentionBlock   — One modality (query) attends to all others (key/value)
    HierarchicalFusionBlock    — Combines intra + cross-modal attention in one block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward block with GELU activation.

    Pre-norm design (LayerNorm before sub-layers) is used throughout for
    more stable training, especially important with multi-modal inputs
    that may have different scale distributions.

    Args:
        d_model  (int): Input/output dimensionality.
        ff_dim   (int): Hidden dimensionality (typically 2× or 4× d_model).
        dropout  (float): Dropout rate.
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# INTRA-MODAL TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class IntraModalTransformer(nn.Module):
    """
    Transformer encoder applied independently within a single modality.

    Purpose:
        Before cross-modal interaction, each modality should first build its
        own internal context. For example, WSI tokens should understand
        spatial relationships between tissue regions; genomic tokens should
        capture co-expression patterns.

    Design:
        Pre-norm Transformer blocks (LayerNorm → Attention → Residual).
        Multiple layers allow hierarchical feature extraction.

    Args:
        d_model   (int): Token dimensionality.
        n_heads   (int): Number of attention heads.
        n_layers  (int): Number of stacked Transformer blocks.
        ff_dim    (int): Feed-forward hidden dimension.
        dropout   (float): Dropout rate.

    Input:  (B, T, d_model)
    Output: (B, T, d_model)    — same shape, enriched representations
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn':  nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm2': nn.LayerNorm(d_model),
                'ff':    FeedForward(d_model, ff_dim, dropout)
            })
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x                (torch.Tensor): Token sequence (B, T, d_model).
            key_padding_mask (torch.Tensor, optional): (B, T), True = ignore.

        Returns:
            torch.Tensor: Updated token sequence (B, T, d_model).
        """
        for layer in self.layers:
            # Pre-norm self-attention + residual
            normed = layer['norm1'](x)
            attn_out, _ = layer['attn'](
                normed, normed, normed,
                key_padding_mask=key_padding_mask
            )
            x = x + self.dropout(attn_out)

            # Pre-norm feed-forward + residual
            x = x + self.dropout(layer['ff'](layer['norm2'](x)))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MODAL ATTENTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal attention: one modality (query) attends to all other modalities
    (key and value), enabling targeted information retrieval across modalities.

    Intuition:
        - Genomic tokens can attend to WSI regions that express certain genes
        - WSI tokens can attend to clinical features that explain tissue patterns
        - MRI tokens can attend to genomic markers that correlate with morphology

    Design:
        query  = tokens from the current modality being updated
        key    = concatenation of all other modality tokens
        value  = concatenation of all other modality tokens

        Cross-attention output is added back to query via residual connection.
        Followed by a feed-forward block with pre-norm.

    Args:
        d_model  (int): Token dimensionality.
        n_heads  (int): Number of attention heads.
        ff_dim   (int): Feed-forward hidden dim.
        dropout  (float): Dropout rate.

    Input:
        query   (B, T_q, d_model)
        context (B, T_ctx, d_model)  — all other modalities concatenated

    Output:
        updated query (B, T_q, d_model)
        attn_weights  (B, T_q, T_ctx)  — interpretable attention maps
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query        (torch.Tensor): Query modality tokens (B, T_q, d_model).
            context      (torch.Tensor): Other modalities concatenated (B, T_ctx, d_model).
            context_mask (torch.Tensor, optional): Padding mask for context (B, T_ctx).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated query tokens (B, T_q, d_model)
                - Attention weights    (B, T_q, T_ctx)  — averaged over heads
        """
        # Pre-norm cross-attention
        q_normed   = self.norm_q(query)
        ctx_normed = self.norm_ctx(context)

        attn_out, attn_weights = self.cross_attn(
            q_normed,
            ctx_normed,
            ctx_normed,
            key_padding_mask=context_mask,
            need_weights=True,
            average_attn_weights=True   # Average over heads for interpretability
        )
        query = query + self.dropout(attn_out)

        # Pre-norm feed-forward
        query = query + self.dropout(self.ff(self.norm_ff(query)))

        return query, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL FUSION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalFusionBlock(nn.Module):
    """
    One complete HCMT hierarchical fusion block:

        Stage 1 (Intra-modal):   Each modality's tokens self-attend independently.
        Stage 2 (Cross-modal):   Each modality's tokens attend to all others.

    Multiple fusion blocks can be stacked for deeper interaction
    (controlled by n_fusion_layers in the config).

    Args:
        d_model         (int): Token dimensionality.
        n_heads         (int): Attention heads.
        n_intra_layers  (int): Intra-modal transformer layers.
        ff_dim          (int): Feed-forward hidden dimension.
        dropout         (float): Dropout rate.
        modality_names  (list): Names of the modalities (for dict I/O).
    """

    MODALITY_NAMES = ['wsi', 'genomics', 'radiology', 'clinical']

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_intra_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Stage 1: Independent intra-modal transformer for each modality
        self.intra = nn.ModuleDict({
            m: IntraModalTransformer(d_model, n_heads, n_intra_layers, ff_dim, dropout)
            for m in self.MODALITY_NAMES
        })

        # Stage 2: Cross-modal attention for each modality
        self.cross = nn.ModuleDict({
            m: CrossModalAttentionBlock(d_model, n_heads, ff_dim, dropout)
            for m in self.MODALITY_NAMES
        })

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            modalities (dict):
                Keys: modality names
                Values: token tensors (B, T_m, d_model) or None if absent

        Returns:
            Tuple:
                - updated_modalities (dict): Same structure, updated representations
                - attn_weights       (dict): Cross-modal attention weights per modality
        """
        # ── Stage 1: Intra-modal self-attention ──────────────────────────────
        intra_out = {}
        for m in self.MODALITY_NAMES:
            if modalities.get(m) is not None:
                intra_out[m] = self.intra[m](modalities[m])
            else:
                intra_out[m] = None

        # ── Stage 2: Cross-modal attention ───────────────────────────────────
        cross_out = {}
        attn_weights = {}

        for m in self.MODALITY_NAMES:
            if intra_out[m] is None:
                cross_out[m] = None
                continue

            # Build context = concatenation of all OTHER available modalities
            context_parts = [
                intra_out[other]
                for other in self.MODALITY_NAMES
                if other != m and intra_out[other] is not None
            ]

            if not context_parts:
                # Only one modality available — skip cross-modal (no context)
                cross_out[m] = intra_out[m]
                attn_weights[m] = None
                continue

            context = torch.cat(context_parts, dim=1)   # (B, sum_T_others, d_model)

            updated, weights = self.cross[m](intra_out[m], context)
            cross_out[m] = updated
            attn_weights[m] = weights                    # (B, T_m, T_ctx)

        return cross_out, attn_weights
