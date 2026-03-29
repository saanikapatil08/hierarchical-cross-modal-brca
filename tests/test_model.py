"""
tests/test_model.py
===================
Full model integration tests and smoke tests.

Run: pytest tests/test_model.py -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hcmt import HCMTClassifier, PAM50_CLASSES


@pytest.fixture
def small_model():
    """Tiny model for fast tests."""
    return HCMTClassifier(
        wsi_feat_dim=64,
        n_genes=200,
        n_clinical_feat=16,
        d_model=32,
        n_heads=4,
        n_intra_layers=1,
        n_fusion_layers=1,
        ff_dim=64,
        n_classes=5,
        dropout=0.0,
        radiology_cnn_channels=[8, 16],
        radiology_pool_size=(2, 2, 2),
        genomics_n_tokens=16,
        clinical_n_tokens=8,
    )


B = 2


class TestHCMTModel:

    def test_forward_all_modalities(self, small_model):
        """Full forward pass with all 4 modalities."""
        wsi  = torch.randn(B, 32, 64)
        gen  = torch.randn(B, 200)
        rad  = torch.randn(B, 1, 8, 16, 16)
        clin = torch.randn(B, 16)

        logits, gate_w, attn = small_model(wsi, gen, rad, clin)

        assert logits.shape  == (B, 5),  f"logits: {logits.shape}"
        assert gate_w.shape  == (B, 4),  f"gate_w: {gate_w.shape}"
        assert isinstance(attn, dict)

    def test_gate_weights_sum_to_one(self, small_model):
        """Gate weights must sum to 1 (softmax output)."""
        wsi  = torch.randn(B, 32, 64)
        gen  = torch.randn(B, 200)
        rad  = torch.randn(B, 1, 8, 16, 16)
        clin = torch.randn(B, 16)

        _, gate_w, _ = small_model(wsi, gen, rad, clin)
        sums = gate_w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"Gate weights don't sum to 1: {sums}"

    def test_forward_missing_radiology(self, small_model):
        """Should work when radiology is None."""
        wsi  = torch.randn(B, 32, 64)
        gen  = torch.randn(B, 200)
        clin = torch.randn(B, 16)

        logits, gate_w, _ = small_model(wsi, gen, None, clin)
        assert logits.shape == (B, 5)
        # Radiology gate weight should be near 0 (masked)
        assert gate_w[:, 2].max().item() < 0.01

    def test_forward_single_modality(self, small_model):
        """Should work with only one modality."""
        gen = torch.randn(B, 200)
        logits, gate_w, _ = small_model(None, gen, None, None)
        assert logits.shape == (B, 5)

    def test_output_no_nan(self, small_model):
        """No NaN or Inf in outputs."""
        wsi  = torch.randn(B, 32, 64)
        gen  = torch.randn(B, 200)
        rad  = torch.randn(B, 1, 8, 16, 16)
        clin = torch.randn(B, 16)

        logits, gate_w, _ = small_model(wsi, gen, rad, clin)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"
        assert not torch.isnan(gate_w).any(), "NaN in gate weights"

    def test_gradient_flows(self, small_model):
        """Loss backward pass should produce gradients."""
        wsi  = torch.randn(B, 32, 64)
        gen  = torch.randn(B, 200)
        clin = torch.randn(B, 16)

        logits, _, _ = small_model(wsi, gen, None, clin)
        loss = logits.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        grads = [p.grad for p in small_model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"

    def test_ablation_no_cross_attn(self):
        """Ablation: disabling cross-modal attention should still produce valid output."""
        model = HCMTClassifier(
            wsi_feat_dim=64, n_genes=200, n_clinical_feat=16,
            d_model=32, n_heads=4, n_intra_layers=1, n_fusion_layers=1, ff_dim=64,
            radiology_cnn_channels=[8,16], radiology_pool_size=(2,2,2),
            genomics_n_tokens=16, clinical_n_tokens=8,
            disable_cross_attn=True
        )
        wsi  = torch.randn(B, 32, 64)
        gen  = torch.randn(B, 200)
        logits, _, _ = model(wsi, gen, None, None)
        assert logits.shape == (B, 5)

    def test_param_count_structure(self, small_model):
        """Parameter count breakdown should have all components."""
        counts = small_model.get_param_count()
        assert 'total' in counts
        assert 'wsi_encoder' in counts
        assert 'fusion_blocks' in counts
        assert counts['total'] > 0

    def test_n_classes_matches_output(self):
        """Output dimension should match n_classes."""
        for n_cls in [2, 3, 5, 10]:
            model = HCMTClassifier(
                wsi_feat_dim=64, n_genes=100, n_clinical_feat=8,
                d_model=32, n_heads=4, n_intra_layers=1, n_fusion_layers=1,
                ff_dim=64, n_classes=n_cls,
                radiology_cnn_channels=[8], radiology_pool_size=(2,2,2),
                genomics_n_tokens=8, clinical_n_tokens=4,
            )
            gen = torch.randn(2, 100)
            logits, _, _ = model(None, gen, None, None)
            assert logits.shape[-1] == n_cls


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
