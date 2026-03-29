"""
src/models/hcmt.py
==================
Full HCMT model — Hierarchical Cross-Modal Transformer
for Breast Cancer Subtype Classification.

This module assembles all components:
    WSIEncoder + GenomicsEncoder + RadiologyEncoder + ClinicalEncoder
    → HierarchicalFusionBlock × n_fusion_layers
    → GatedModalityFusion
    → MLP Classifier

Usage:
    model = HCMTClassifier.from_config(cfg)
    logits, gate_weights, attn_weights = model(wsi, genomics, radiology, clinical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple

from .encoders import (
    WSIEncoder,
    GenomicsEncoder,
    RadiologyEncoder,
    ClinicalEncoder,
    MissingModalityHandler
)
from .attention import HierarchicalFusionBlock
from .fusion import GatedModalityFusion


# PAM50 class definitions for reference
PAM50_CLASSES = {
    0: 'LumA',
    1: 'LumB',
    2: 'Her2',
    3: 'Basal',   # Triple-Negative
    4: 'Normal'
}
PAM50_TO_IDX = {v: k for k, v in PAM50_CLASSES.items()}
IDX_TO_PAM50 = {v: k for k, v in PAM50_TO_IDX.items()}


class HCMTClassifier(nn.Module):
    """
    Hierarchical Cross-Modal Transformer (HCMT) Classifier.

    Full pipeline from raw multi-modal inputs to PAM50 subtype predictions.

    Architecture overview:
        ┌─────────────────────────────────────────────┐
        │           MODALITY ENCODERS                 │
        │  WSI │ Genomics │ Radiology │ Clinical      │
        └──────────────────┬──────────────────────────┘
                           │ token sequences
        ┌──────────────────▼──────────────────────────┐
        │    HIERARCHICAL FUSION BLOCKS × L            │
        │  [Intra-modal Attn] → [Cross-modal Attn]    │
        └──────────────────┬──────────────────────────┘
                           │ enriched tokens
        ┌──────────────────▼──────────────────────────┐
        │         GATED MODALITY FUSION                │
        │   Attention Pool → Soft Gate → Weighted Sum  │
        └──────────────────┬──────────────────────────┘
                           │ fused vector (B, d_model)
        ┌──────────────────▼──────────────────────────┐
        │           MLP CLASSIFIER                     │
        │         → 5 PAM50 logits                    │
        └─────────────────────────────────────────────┘

    Args:
        wsi_feat_dim      (int): Dimension of pre-extracted WSI patch features.
        n_genes           (int): Number of RNA-seq genes.
        n_clinical_feat   (int): Number of clinical input features.
        d_model           (int): Hidden dimension throughout.
        n_heads           (int): Number of attention heads.
        n_intra_layers    (int): Intra-modal transformer layers per block.
        n_fusion_layers   (int): Number of stacked HierarchicalFusionBlocks.
        ff_dim            (int): Feed-forward hidden dimension.
        n_classes         (int): Number of output classes (5 for PAM50).
        dropout           (float): Dropout rate.
        disable_cross_attn(bool): Ablation — disable cross-modal attention.
        disable_gate      (bool): Ablation — disable gated fusion (use mean instead).

    Forward Args:
        wsi_feats  (B, N_patches, wsi_feat_dim) — patch features
        genomics   (B, n_genes)                 — RNA-seq expression
        radiology  (B, 1, D, H, W)              — MRI/CT volume
        clinical   (B, n_clinical_feat)          — tabular features

        Any of the above can be None if that modality is unavailable.

    Returns:
        logits       (B, n_classes)     — raw classification logits
        gate_weights (B, n_modalities)  — per-modality contribution scores
        attn_weights (dict)             — cross-modal attention maps per modality
    """

    def __init__(
        self,
        wsi_feat_dim: int = 1024,
        n_genes: int = 20531,
        n_clinical_feat: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_intra_layers: int = 2,
        n_fusion_layers: int = 2,
        ff_dim: int = 512,
        n_classes: int = 5,
        dropout: float = 0.1,
        # Ablation flags
        disable_cross_attn: bool = False,
        disable_gate: bool = False,
        # Encoder-specific
        wsi_n_proj_layers: int = 2,
        genomics_n_tokens: int = 64,
        radiology_cnn_channels: list = None,
        radiology_pool_size: tuple = (4, 4, 4),
        clinical_n_tokens: int = 16,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes
        self.disable_cross_attn = disable_cross_attn
        self.disable_gate = disable_gate

        if radiology_cnn_channels is None:
            radiology_cnn_channels = [32, 64, 128]

        # ── Modality Encoders ─────────────────────────────────────────────────
        self.wsi_encoder = WSIEncoder(
            patch_feat_dim=wsi_feat_dim,
            d_model=d_model,
            n_proj_layers=wsi_n_proj_layers,
            dropout=dropout
        )
        self.genomics_encoder = GenomicsEncoder(
            n_genes=n_genes,
            d_model=d_model,
            n_tokens=genomics_n_tokens,
            dropout=dropout
        )
        self.radiology_encoder = RadiologyEncoder(
            d_model=d_model,
            cnn_channels=radiology_cnn_channels,
            pool_size=radiology_pool_size,
            dropout=dropout
        )
        self.clinical_encoder = ClinicalEncoder(
            n_features=n_clinical_feat,
            d_model=d_model,
            n_tokens=clinical_n_tokens,
            dropout=dropout
        )

        # ── Missing Modality Handler ──────────────────────────────────────────
        # Compute default token counts for mask tokens
        pool_vol = radiology_pool_size[0] * radiology_pool_size[1] * radiology_pool_size[2]
        self.missing_handler = MissingModalityHandler(
            d_model=d_model,
            default_token_counts={
                'wsi': 257,                   # 256 patches + CLS (adjust as needed)
                'genomics': genomics_n_tokens,
                'radiology': pool_vol,
                'clinical': clinical_n_tokens,
            }
        )

        # ── Hierarchical Fusion Blocks ────────────────────────────────────────
        self.fusion_blocks = nn.ModuleList([
            HierarchicalFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_intra_layers=n_intra_layers,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(n_fusion_layers)
        ])

        # ── Gated Fusion ──────────────────────────────────────────────────────
        self.gated_fusion = GatedModalityFusion(
            d_model=d_model,
            n_modalities=4,
            pool_type='attention',
            dropout=dropout
        )

        # ── Classifier Head ───────────────────────────────────────────────────
        classifier_hidden = d_model // 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes)
        )

        # ── Weight Initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layers with truncated normal initialization,
        as recommended for Transformer-based architectures.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_modalities(
        self,
        wsi_feats: Optional[torch.Tensor],
        genomics: Optional[torch.Tensor],
        radiology: Optional[torch.Tensor],
        clinical: Optional[torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Run each available modality through its encoder.

        Returns dict with None for missing modalities.
        """
        return {
            'wsi':       self.wsi_encoder(wsi_feats)       if wsi_feats  is not None else None,
            'genomics':  self.genomics_encoder(genomics)   if genomics   is not None else None,
            'radiology': self.radiology_encoder(radiology) if radiology  is not None else None,
            'clinical':  self.clinical_encoder(clinical)   if clinical   is not None else None,
        }

    def forward(
        self,
        wsi_feats:  Optional[torch.Tensor] = None,
        genomics:   Optional[torch.Tensor] = None,
        radiology:  Optional[torch.Tensor] = None,
        clinical:   Optional[torch.Tensor] = None,
        available:  Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            wsi_feats  (tensor, optional): (B, N_patches, wsi_feat_dim)
            genomics   (tensor, optional): (B, n_genes)
            radiology  (tensor, optional): (B, 1, D, H, W)
            clinical   (tensor, optional): (B, n_clinical_feat)
            available  (dict, optional):   Per-sample availability masks.
                                           Keys: modality names, values: (B,) bool tensors.
                                           If None, all provided modalities assumed available.

        Returns:
            logits       (B, n_classes)     — classification logits
            gate_weights (B, 4)             — per-modality soft weights
            attn_weights (dict)             — cross-modal attn maps, keys = modality names
        """
        # ── Step 1: Encode each modality ─────────────────────────────────────
        modalities = self._encode_modalities(wsi_feats, genomics, radiology, clinical)

        # ── Step 2: Handle missing modalities ────────────────────────────────
        if available is not None:
            modalities = self.missing_handler(modalities, available)

        # ── Step 3: Stacked Hierarchical Fusion Blocks ───────────────────────
        all_attn_weights = {}
        for block_idx, block in enumerate(self.fusion_blocks):
            if self.disable_cross_attn:
                # Ablation: only run intra-modal attention, skip cross-modal
                from .attention import IntraModalTransformer
                updated = {}
                for m, tokens in modalities.items():
                    if tokens is not None:
                        updated[m] = block.intra[m](tokens)
                    else:
                        updated[m] = None
                modalities = updated
                all_attn_weights = {}
            else:
                modalities, attn = block(modalities)
                all_attn_weights = attn   # Keep last block's attention

        # ── Step 4: Gated Modality Fusion ────────────────────────────────────
        if self.disable_gate:
            # Ablation: simple mean pooling + equal weighting (no gate)
            pooled = []
            for m, tokens in modalities.items():
                if tokens is not None:
                    pooled.append(tokens.mean(dim=1))
            fused = torch.stack(pooled, dim=1).mean(dim=1)
            gate_weights = torch.ones(fused.size(0), 4, device=fused.device) / 4
        else:
            fused, gate_weights = self.gated_fusion(modalities)

        # ── Step 5: Classify ─────────────────────────────────────────────────
        logits = self.classifier(fused)    # (B, n_classes)

        return logits, gate_weights, all_attn_weights

    @classmethod
    def from_config(cls, cfg: DictConfig) -> 'HCMTClassifier':
        """
        Instantiate HCMT from an OmegaConf config object.

        Args:
            cfg: OmegaConf DictConfig loaded from configs/default.yaml

        Returns:
            Initialized HCMTClassifier.

        Example:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('configs/default.yaml')
            model = HCMTClassifier.from_config(cfg)
        """
        m = cfg.model
        a = cfg.get('ablation', {})

        return cls(
            wsi_feat_dim=cfg.data.wsi_feat_dim,
            n_genes=cfg.data.n_genes,
            n_clinical_feat=cfg.data.n_clinical_features,
            d_model=m.d_model,
            n_heads=m.n_heads,
            n_intra_layers=m.intra_modal.n_layers,
            n_fusion_layers=m.cross_modal.n_layers,
            ff_dim=m.intra_modal.ff_dim,
            n_classes=m.n_classes,
            dropout=m.dropout,
            disable_cross_attn=a.get('disable_cross_modal_attn', False),
            disable_gate=a.get('disable_gated_fusion', False),
            wsi_n_proj_layers=m.encoders.wsi.n_proj_layers,
            genomics_n_tokens=m.encoders.genomics.n_tokens,
            radiology_cnn_channels=list(m.encoders.radiology.cnn_channels),
            radiology_pool_size=tuple(m.encoders.radiology.pool_size),
            clinical_n_tokens=m.encoders.clinical.n_tokens,
        )

    def get_param_count(self) -> Dict[str, int]:
        """Returns parameter counts broken down by component."""
        components = {
            'wsi_encoder':     self.wsi_encoder,
            'genomics_encoder': self.genomics_encoder,
            'radiology_encoder': self.radiology_encoder,
            'clinical_encoder':  self.clinical_encoder,
            'fusion_blocks':    self.fusion_blocks,
            'gated_fusion':     self.gated_fusion,
            'classifier':       self.classifier,
        }
        counts = {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in components.items()
        }
        counts['total'] = sum(counts.values())
        return counts
