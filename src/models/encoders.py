"""
src/models/encoders.py
======================
Modality-specific encoder modules for HCMT.

Each encoder takes raw modality input and produces a sequence of d_model-dimensional
tokens suitable for downstream transformer processing.

Encoders:
    WSIEncoder       — Whole Slide Image patch-feature encoder
    GenomicsEncoder  — RNA-seq / gene expression encoder
    RadiologyEncoder — 3D MRI/CT volume encoder
    ClinicalEncoder  — Tabular EHR/clinical feature encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# WSI ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class WSIEncoder(nn.Module):
    """
    Encodes Whole Slide Image (WSI) pre-extracted patch features.

    Workflow:
        1. Accept pre-computed patch embeddings from a pathology foundation model
           (e.g., CONCH, UNI, PLIP, CTRANSPATH).
        2. Project to d_model via a 2-layer MLP.
        3. Prepend a learnable [CLS] token whose representation aggregates
           the global slide-level context.

    Why pre-extracted features?
        Processing raw gigapixel slides (50,000 × 50,000 px) is infeasible
        end-to-end. CLAM extracts ~256×256 px patches; a foundation model
        encodes each patch to a fixed vector offline.

    Args:
        patch_feat_dim (int): Dimensionality of input patch features (e.g., 1024 for CONCH).
        d_model (int): Output token dimensionality.
        n_proj_layers (int): Number of MLP projection layers (1 or 2).
        dropout (float): Dropout rate.

    Input shape:  (B, N_patches, patch_feat_dim)
    Output shape: (B, N_patches + 1, d_model)   — +1 for [CLS] token
    """

    def __init__(
        self,
        patch_feat_dim: int = 1024,
        d_model: int = 256,
        n_proj_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Build projection MLP
        layers = []
        in_dim = patch_feat_dim
        for i in range(n_proj_layers):
            out_dim = d_model if i == n_proj_layers - 1 else d_model * 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.proj = nn.Sequential(*layers)

        # Learnable [CLS] token — summary of the entire slide
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional embedding (applied before CLS)
        # Note: We use a simple learned positional bias rather than fixed sinusoidal
        # because patch order is arbitrary (MIL assumption).
        self.pos_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x    (torch.Tensor): Patch features, shape (B, N, patch_feat_dim).
            mask (torch.Tensor, optional): Boolean mask for padded patches, shape (B, N).
                                           True = padded (ignore this patch).

        Returns:
            torch.Tensor: Token sequence (B, N+1, d_model).
                          Index 0 is the [CLS] token.
        """
        B, N, _ = x.shape

        # Project patches → d_model
        x = self.proj(x)                              # (B, N, d_model)
        x = self.pos_drop(x)

        # Prepend [CLS] token
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1)                # (B, N+1, d_model)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# GENOMICS ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class GenomicsEncoder(nn.Module):
    """
    Encodes high-dimensional RNA-seq gene expression vectors.

    The challenge: TCGA RNA-seq has ~20,531 genes — too many for direct attention.

    Solution — Factored Tokenization:
        Compress the gene expression vector into a small, fixed number of learned
        "genomic tokens" via a bottleneck MLP. Each token can be seen as summarizing
        a gene program or pathway.

    Optional: Gene grouping by pathway (MSigDB hallmarks) for biologically
    meaningful tokenization (set n_tokens to number of pathways, e.g., 50).

    Args:
        n_genes   (int): Number of input genes (default: 20531 for TCGA).
        d_model   (int): Output token dimensionality.
        n_tokens  (int): Number of output genomic tokens.
        hidden_dim(int): Hidden dim in the bottleneck MLP.
        dropout   (float): Dropout rate.

    Input shape:  (B, n_genes)
    Output shape: (B, n_tokens, d_model)
    """

    def __init__(
        self,
        n_genes: int = 20531,
        d_model: int = 256,
        n_tokens: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Bottleneck MLP: n_genes → hidden → n_tokens * d_model
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tokens * d_model),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Normalized gene expression, shape (B, n_genes).
                              Typically log1p-normalized TPM values.

        Returns:
            torch.Tensor: Genomic token sequence, shape (B, n_tokens, d_model).
        """
        B = x.size(0)
        x = self.encoder(x)                           # (B, n_tokens * d_model)
        x = x.view(B, self.n_tokens, self.d_model)    # (B, n_tokens, d_model)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# RADIOLOGY ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class RadiologyEncoder(nn.Module):
    """
    Encodes 3D MRI / CT volumes using a hierarchical 3D CNN backbone.

    Architecture:
        3× Conv3D blocks (stride 2 → progressive downsampling)
        → AdaptiveAvgPool3D to fixed spatial size (4×4×4)
        → Flatten spatial dims → token sequence
        → Linear projection to d_model

    Why CNN instead of 3D ViT?
        - 3D ViTs require very large pre-training datasets
        - 3D CNNs are parameter-efficient and work well on limited data
        - Spatial inductive bias is useful for volumetric anatomy

    Args:
        d_model      (int): Output token dimensionality.
        cnn_channels (list): Channels for each conv block.
        pool_size    (tuple): Spatial size after adaptive pooling (D, H, W).
        dropout      (float): Dropout rate.

    Input shape:  (B, 1, D, H, W)   — single-channel (grayscale MRI)
    Output shape: (B, D*H*W, d_model)  i.e., (B, 64, d_model) for pool_size=(4,4,4)
    """

    def __init__(
        self,
        d_model: int = 256,
        cnn_channels: list = [32, 64, 128],
        pool_size: tuple = (4, 4, 4),
        dropout: float = 0.1
    ):
        super().__init__()

        # Build 3D CNN backbone
        layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(pool_size)

        # Token count = D * H * W after pooling
        n_tokens = pool_size[0] * pool_size[1] * pool_size[2]

        self.proj = nn.Sequential(
            nn.Linear(cnn_channels[-1], d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self._n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Volumetric image, shape (B, 1, D, H, W).

        Returns:
            torch.Tensor: Token sequence, shape (B, n_tokens, d_model).
        """
        x = self.backbone(x)              # (B, C, D', H', W')
        x = self.pool(x)                  # (B, C, pool_D, pool_H, pool_W)

        B, C, D, H, W = x.shape
        x = x.view(B, C, D * H * W)      # (B, C, n_tokens)
        x = rearrange(x, 'b c t -> b t c')  # (B, n_tokens, C)
        x = self.proj(x)                  # (B, n_tokens, d_model)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalEncoder(nn.Module):
    """
    Encodes tabular clinical / EHR features.

    Clinical features typically include:
        - Age at diagnosis (continuous, normalized)
        - Tumor size (continuous)
        - Lymph node status (ordinal)
        - Histological grade (ordinal: 1, 2, 3)
        - ER / PR / HER2 IHC status (binary)
        - Stage (categorical, one-hot)
        - Menopausal status (binary)

    These are concatenated into a flat feature vector of size n_features,
    then projected to n_tokens learned clinical tokens.

    Args:
        n_features (int): Total number of clinical input features.
        d_model    (int): Output token dimensionality.
        n_tokens   (int): Number of clinical tokens.
        hidden_dim (int): MLP hidden dimension.
        dropout    (float): Dropout rate.

    Input shape:  (B, n_features)
    Output shape: (B, n_tokens, d_model)
    """

    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 256,
        n_tokens: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tokens * d_model),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Clinical feature vector, shape (B, n_features).

        Returns:
            torch.Tensor: Clinical token sequence, shape (B, n_tokens, d_model).
        """
        B = x.size(0)
        x = self.encoder(x)
        x = x.view(B, self.n_tokens, self.d_model)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING MODALITY HANDLER
# ─────────────────────────────────────────────────────────────────────────────

class MissingModalityHandler(nn.Module):
    """
    Handles cases where one or more modalities are absent for a patient.

    Strategy: Learned mask tokens.
        When a modality is missing, replace its token sequence with a learned
        "mask" embedding. This teaches the model what "absence of data" looks
        like, rather than crashing or using zeros.

    This is critical for real clinical deployment where not every patient has
    all four modalities.

    Args:
        d_model      (int): Token dimensionality.
        modality_names (list): Names of modalities.
        default_token_counts (dict): Default token counts per modality
                                     (used to set mask token sequence length).
    """

    def __init__(
        self,
        d_model: int = 256,
        modality_names: list = None,
        default_token_counts: dict = None
    ):
        super().__init__()
        if modality_names is None:
            modality_names = ['wsi', 'genomics', 'radiology', 'clinical']
        if default_token_counts is None:
            default_token_counts = {
                'wsi': 257,        # 256 patches + CLS
                'genomics': 64,
                'radiology': 64,
                'clinical': 16
            }

        self.modality_names = modality_names
        self.default_token_counts = default_token_counts

        # One learned mask token per modality (broadcast to sequence length)
        self.mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            for m in modality_names
        })

    def forward(
        self,
        modalities: dict,
        available: dict
    ) -> dict:
        """
        Args:
            modalities (dict): Keys are modality names, values are tensors
                               (B, T_m, d_model) or None if missing.
            available  (dict): Keys are modality names, values are bool tensors
                               of shape (B,) — True if available for that sample.

        Returns:
            dict: Same structure as modalities, but missing modalities are
                  filled with learned mask tokens.
        """
        filled = {}
        for m in self.modality_names:
            tokens = modalities.get(m, None)
            avail_mask = available.get(m, None)

            if tokens is None:
                # Entire modality missing for whole batch
                B = next(t.size(0) for t in modalities.values() if t is not None)
                T = self.default_token_counts[m]
                device = self.mask_tokens[m].device
                filled[m] = self.mask_tokens[m].expand(B, T, -1).clone()
            elif avail_mask is not None:
                # Some samples in batch are missing this modality
                B, T, D = tokens.shape
                mask = self.mask_tokens[m].expand(B, T, -1)
                # avail_mask: (B,) → (B, 1, 1) for broadcasting
                a = avail_mask.float().view(B, 1, 1)
                filled[m] = tokens * a + mask * (1 - a)
            else:
                filled[m] = tokens

        return filled
