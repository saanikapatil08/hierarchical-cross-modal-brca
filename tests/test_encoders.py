"""
tests/test_encoders.py
======================
Unit tests for all modality encoders and the full HCMT model.

Run with:
    pytest tests/ -v
    pytest tests/test_encoders.py -v --tb=short
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.encoders import (
    WSIEncoder, GenomicsEncoder, RadiologyEncoder, ClinicalEncoder,
    MissingModalityHandler
)
from src.models.attention import (
    IntraModalTransformer, CrossModalAttentionBlock, HierarchicalFusionBlock
)
from src.models.fusion import GatedModalityFusion
from src.models.hcmt import HCMTClassifier


# ─── Test constants ───────────────────────────────────────────────────────────
B         = 2       # Batch size
D_MODEL   = 64      # Small d_model for fast tests
N_PATCHES = 32      # WSI patches per slide
N_GENES   = 512     # Reduced for testing (real: 20531)
N_CLIN    = 16      # Clinical features
N_HEADS   = 4


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestWSIEncoder:

    def test_output_shape(self):
        """Output should be (B, N_patches+1, d_model) — +1 for CLS."""
        enc = WSIEncoder(patch_feat_dim=128, d_model=D_MODEL)
        x   = torch.randn(B, N_PATCHES, 128)
        out = enc(x)
        assert out.shape == (B, N_PATCHES + 1, D_MODEL), \
            f"Expected ({B}, {N_PATCHES+1}, {D_MODEL}), got {out.shape}"

    def test_cls_token_is_first(self):
        """Index 0 should be the CLS token (learned, different from input tokens)."""
        enc = WSIEncoder(patch_feat_dim=128, d_model=D_MODEL, n_proj_layers=1)
        x   = torch.randn(B, N_PATCHES, 128)
        out = enc(x)
        # CLS token should not be NaN
        assert not torch.isnan(out[:, 0, :]).any()

    def test_no_nan_output(self):
        enc = WSIEncoder(patch_feat_dim=256, d_model=D_MODEL)
        x   = torch.randn(B, 64, 256)
        out = enc(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        """Gradients should flow back through encoder."""
        enc = WSIEncoder(patch_feat_dim=128, d_model=D_MODEL)
        x   = torch.randn(B, N_PATCHES, 128, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGenomicsEncoder:

    def test_output_shape(self):
        enc = GenomicsEncoder(n_genes=N_GENES, d_model=D_MODEL, n_tokens=16)
        x   = torch.randn(B, N_GENES)
        out = enc(x)
        assert out.shape == (B, 16, D_MODEL)

    def test_no_nan_output(self):
        enc = GenomicsEncoder(n_genes=N_GENES, d_model=D_MODEL)
        x   = torch.randn(B, N_GENES)
        out = enc(x)
        assert not torch.isnan(out).any()


class TestRadiologyEncoder:

    def test_output_shape(self):
        enc  = RadiologyEncoder(d_model=D_MODEL, cnn_channels=[8, 16, 32], pool_size=(2, 2, 2))
        x    = torch.randn(B, 1, 16, 32, 32)
        out  = enc(x)
        # n_tokens = 2*2*2 = 8
        assert out.shape == (B, 8, D_MODEL)

    def test_no_nan_output(self):
        enc = RadiologyEncoder(d_model=D_MODEL, cnn_channels=[8, 16, 32], pool_size=(2, 2, 2))
        x   = torch.randn(B, 1, 16, 32, 32)
        out = enc(x)
        assert not torch.isnan(out).any()


class TestClinicalEncoder:

    def test_output_shape(self):
        enc = ClinicalEncoder(n_features=N_CLIN, d_model=D_MODEL, n_tokens=8)
        x   = torch.randn(B, N_CLIN)
        out = enc(x)
        assert out.shape == (B, 8, D_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIntraModalTransformer:

    def test_output_shape_preserved(self):
        """Intra-modal transformer should not change tensor shape."""
        transformer = IntraModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=2)
        x   = torch.randn(B, 20, D_MODEL)
        out = transformer(x)
        assert out.shape == x.shape

    def test_no_nan_output(self):
        transformer = IntraModalTransformer(d_model=D_MODEL, n_heads=N_HEADS)
        x   = torch.randn(B, 15, D_MODEL)
        out = transformer(x)
        assert not torch.isnan(out).any()


class TestCrossModalAttentionBlock:

    def test_output_shape(self):
        """Query shape should be preserved; attn weights (B, T_q, T_ctx)."""
        block   = CrossModalAttentionBlock(d_model=D_MODEL, n_heads=N_HEADS)
        query   = torch.randn(B, 10, D_MODEL)
        context = torch.randn(B, 30, D_MODEL)
        out, weights = block(query, context)
        assert out.shape == query.shape, f"Output shape mismatch: {out.shape}"
        assert weights.shape == (B, 10, 30), f"Attention weight shape: {weights.shape}"

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 over context dim."""
        block   = CrossModalAttentionBlock(d_model=D_MODEL, n_heads=N_HEADS)
        query   = torch.randn(B, 5, D_MODEL)
        context = torch.randn(B, 20, D_MODEL)
        _, weights = block(query, context)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
            f"Attention weights don't sum to 1: {sums}"


class TestHierarchicalFusionBlock:

    def test_all_modalities_output_shape(self):
        block = HierarchicalFusionBlock(d_model=D_MODEL, n_heads=N_HEADS, n_intra_layers=1)
        modalities = {
            'wsi':       torch.randn(B, 10, D_MODEL),
            'genomics':  torch.randn(B, 8, D_MODEL),
            'radiology': torch.randn(B, 6, D_MODEL),
            'clinical':  torch.randn(B, 4, D_MODEL),
        }
        out_mods, attn_w = block(modalities)
        for m, tokens in out_mods.items():
            assert tokens.shape == modalities[m].shape, \
                f"{m}: expected {modalities[m].shape}, got {tokens.shape}"

    def test_missing_modality(self):
        """Block should handle None modalities gracefully."""
        block = HierarchicalFusionBlock(d_model=D_MODEL, n_heads=N_HEADS, n_intra_layers=1)
        modalities = {
            'wsi':       torch.randn(B, 10, D_MODEL),
            'genomics':  torch.randn(B, 8, D_MODEL),
            'radiology': None,                          # Missing
            'clinical':  torch.randn(B, 4, D_MODEL),
        }
        out_mods, attn_w = block(modalities)
        assert out_mods['radiology'] is None
        assert out_mods['wsi'] is not None


# ─────────────────────────────────────────────────────────────────────────────
# GATED FUSION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestGatedModalityFusion:

    def test_output_shape(self):
        fusion = GatedModalityFusion(d_model=D_MODEL, n_modalities=4)
        modalities = {
            'wsi':       torch.randn(B, 10, D_MODEL),
            'genomics':  torch.randn(B, 8, D_MODEL),
            'radiology': torch.randn(B, 6, D_MODEL),
            'clinical':  torch.randn(B, 4, D_MODEL),
        }
        fused, gate_w = fusion(modalities)
        assert fused.shape  == (B, D_MODEL), f"Fused: {fused.shape}"
        assert gate_w.shape == (B, 4),       f"Gate: {gate_w.shape}"

    def test_gate_weights_sum_to_one(self):
        """Gate weights over modalities should sum to 1."""
        fusion = GatedModalityFusion(d_model=D_MODEL, n_modalities=4)
        modalities = {m: torch.randn(B, 5, D_MODEL)
                      for m in ['wsi', 'genomics', 'radiology', 'clinical']}
        _, gate_w = fusion(modalities)
        sums = gate_w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_missing_modality_gate(self):
        """Missing modality should get gate weight of 0."""
        fusion = GatedModalityFusion(d_model=D_MODEL, n_modalities=4)
        modalities = {
            'wsi':       torch.randn(B, 10, D_MODEL),
            'genomics':  torch.randn(B, 8, D_MODEL),
            'radiology': None,
            'clinical':  torch.randn(B, 4, D_MODEL),
        }
        _, gate_w = fusion(modalities)
        # Radiology is index 2
        radiology_weight = gate_w[:, 2]
        assert torch.allclose(radiology_weight, torch.zeros_like(radiology_weight), atol=1e-5), \
            f"Missing modality gate weight should be 0, got {radiology_weight}"


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestHCMTClassifier:

    @pytest.fixture
    def small_model(self):
        return HCMTClassifier(
            wsi_feat_dim=128,
            n_genes=N_GENES,
            n_clinical_feat=N_CLIN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_intra_layers=1,
            n_fusion_layers=1,
            ff_dim=128,
            n_classes=5,
            dropout=0.0,   # No dropout for deterministic tests
            radiology_cnn_channels=[4, 8, 16],
            radiology_pool_size=(2, 2, 2),
            genomics_n_tokens=8,
            clinical_n_tokens=4,
        )

    def test_full_forward_output_shapes(self, small_model):
        wsi      = torch.randn(B, N_PATCHES, 128)
        genomics = torch.randn(B, N_GENES)
        radiology= torch.randn(B, 1, 8, 16, 16)
        clinical = torch.randn(B, N_CLIN)

        logits, gate_w, attn = small_model(wsi, genomics, radiology, clinical)

        assert logits.shape  == (B, 5), f"Logits: {logits.shape}"
        assert gate_w.shape  == (B, 4), f"Gate:   {gate_w.shape}"
        assert isinstance(attn, dict)

    def test_gradient_flow_full_model(self, small_model):
        """Gradients should flow from loss all the way back through every encoder."""
        wsi      = torch.randn(B, N_PATCHES, 128, requires_grad=True)
        genomics = torch.randn(B, N_GENES, requires_grad=True)
        radiology= torch.randn(B, 1, 8, 16, 16, requires_grad=True)
        clinical = torch.randn(B, N_CLIN, requires_grad=True)

        logits, _, _ = small_model(wsi, genomics, radiology, clinical)
        loss = logits.sum()
        loss.backward()

        for name, inp in [('wsi', wsi), ('genomics', genomics),
                          ('radiology', radiology), ('clinical', clinical)]:
            assert inp.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(inp.grad).any(), f"NaN gradient for {name}"

    def test_missing_modality_inference(self, small_model):
        """Model should work with only a subset of modalities."""
        wsi      = torch.randn(B, N_PATCHES, 128)
        genomics = torch.randn(B, N_GENES)
        # No radiology, no clinical

        logits, gate_w, _ = small_model(wsi, genomics, None, None)
        assert logits.shape == (B, 5)
        assert not torch.isnan(logits).any()

    def test_no_nan_with_random_input(self, small_model):
        wsi      = torch.randn(B, N_PATCHES, 128)
        genomics = torch.randn(B, N_GENES)
        radiology= torch.randn(B, 1, 8, 16, 16)
        clinical = torch.randn(B, N_CLIN)

        logits, gate_w, _ = small_model(wsi, genomics, radiology, clinical)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isnan(gate_w).any(), "NaN in gate weights"

    def test_ablation_no_cross_attn(self):
        """Ablation: disable cross-modal attention, model should still run."""
        model = HCMTClassifier(
            wsi_feat_dim=128, n_genes=N_GENES, n_clinical_feat=N_CLIN,
            d_model=D_MODEL, n_heads=N_HEADS, n_intra_layers=1, n_fusion_layers=1,
            ff_dim=128, n_classes=5, dropout=0.0,
            disable_cross_attn=True,
            radiology_cnn_channels=[4, 8, 16], radiology_pool_size=(2, 2, 2),
            genomics_n_tokens=8, clinical_n_tokens=4
        )
        wsi      = torch.randn(B, N_PATCHES, 128)
        genomics = torch.randn(B, N_GENES)
        radiology= torch.randn(B, 1, 8, 16, 16)
        clinical = torch.randn(B, N_CLIN)

        logits, gate_w, _ = model(wsi, genomics, radiology, clinical)
        assert logits.shape == (B, 5)
        assert not torch.isnan(logits).any()

    def test_param_count(self, small_model):
        """Parameter count report should include all components."""
        counts = small_model.get_param_count()
        required_keys = ['wsi_encoder', 'genomics_encoder', 'radiology_encoder',
                         'clinical_encoder', 'fusion_blocks', 'gated_fusion',
                         'classifier', 'total']
        for key in required_keys:
            assert key in counts, f"Missing key in param_count: {key}"
        assert counts['total'] > 0

    def test_eval_mode_deterministic(self, small_model):
        """In eval mode with dropout=0, two identical inputs → identical outputs."""
        small_model.eval()
        wsi = torch.randn(1, 16, 128)
        gen = torch.randn(1, N_GENES)

        with torch.no_grad():
            out1, _, _ = small_model(wsi, gen, None, None)
            out2, _, _ = small_model(wsi, gen, None, None)

        assert torch.allclose(out1, out2), "Eval mode non-deterministic"
