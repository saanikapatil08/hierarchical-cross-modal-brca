"""
scripts/predict.py
==================
Single-sample inference script for clinical use / demo.

Given one patient's available modalities, returns:
    - Predicted PAM50 subtype + confidence
    - Probability distribution over all 5 subtypes
    - Modality gate weights (which modalities drove the prediction)
    - Attention-based explanation text

Usage:
    # All four modalities
    python scripts/predict.py \
        --checkpoint checkpoints/hcmt_baseline_best.pth \
        --config configs/default.yaml \
        --wsi_features /data/features/TCGA-A1-A0SK.pt \
        --genomics /data/genomics/TCGA-A1-A0SK.pt \
        --radiology /data/radiology/TCGA-A1-A0SK.pt \
        --clinical_csv /data/clinical_one_patient.csv \
        --patient_id TCGA-A1-A0SK

    # WSI + genomics only (missing radiology and clinical)
    python scripts/predict.py \
        --checkpoint checkpoints/hcmt_baseline_best.pth \
        --config configs/default.yaml \
        --wsi_features /data/features/TCGA-A1-A0SK.pt \
        --genomics /data/genomics/TCGA-A1-A0SK.pt
"""

import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hcmt import HCMTClassifier, PAM50_CLASSES, IDX_TO_PAM50
from src.data.dataset import (
    CLINICAL_NUMERIC_COLS, CLINICAL_BINARY_COLS, CLINICAL_ONEHOT_COLS
)
from src.utils.utils import load_config, load_checkpoint, get_logger

logger = get_logger(__name__)


def load_clinical_features(
    clinical_csv: str,
    patient_id: str,
    n_features: int = 32
) -> torch.Tensor:
    """
    Load and encode clinical features from a CSV for a single patient.

    The CSV should contain one row per patient with columns matching
    CLINICAL_NUMERIC_COLS, CLINICAL_BINARY_COLS, and CLINICAL_ONEHOT_COLS.

    Returns:
        Tensor of shape (n_features,).
    """
    df = pd.read_csv(clinical_csv, index_col='patient_id')
    if patient_id not in df.index:
        logger.warning(f"Patient {patient_id} not found in {clinical_csv}. Using zeros.")
        return torch.zeros(n_features)

    row = df.loc[patient_id]
    features = []

    for c in CLINICAL_NUMERIC_COLS:
        features.append(float(row.get(c, 0.0)))

    for c in CLINICAL_BINARY_COLS:
        features.append(float(row.get(c, 0.0)))

    for col, categories in CLINICAL_ONEHOT_COLS.items():
        val = str(row.get(col, 'Unknown'))
        for cat in categories:
            features.append(1.0 if val == cat else 0.0)

    # Pad or truncate to n_features
    while len(features) < n_features:
        features.append(0.0)
    features = features[:n_features]

    return torch.tensor(features, dtype=torch.float32)


def format_prediction(
    patient_id: str,
    probs: np.ndarray,
    gate_weights: np.ndarray,
    modalities_used: list
) -> str:
    """Format prediction results as a readable report."""

    pred_idx  = int(probs.argmax())
    pred_cls  = PAM50_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    lines = [
        f"\n{'='*60}",
        f"  HCMT Prediction Report",
        f"  Patient: {patient_id}",
        f"{'='*60}",
        f"\n  PREDICTED SUBTYPE:  {pred_cls}",
        f"  CONFIDENCE:         {confidence:.1%}",
        f"\n  Probability Distribution:",
    ]

    # Sorted by probability
    sorted_subtypes = sorted(
        zip(PAM50_CLASSES, probs.tolist()),
        key=lambda x: x[1], reverse=True
    )
    for cls, prob in sorted_subtypes:
        bar = '█' * int(prob * 30)
        marker = ' ◄' if cls == pred_cls else ''
        lines.append(f"    {cls:<8}  {bar:<30}  {prob:.3f}{marker}")

    lines.append(f"\n  Modality Contributions (Gate Weights):")
    mod_names = ['WSI', 'Genomics', 'Radiology', 'Clinical']
    for mod, w in zip(mod_names, gate_weights.tolist()):
        used = "✓" if mod.lower().replace(' ', '') in [m.lower() for m in modalities_used] else "✗ (masked)"
        bar  = '▓' * int(w * 30)
        lines.append(f"    {mod:<12}  {bar:<30}  {w:.3f}  {used}")

    lines.append(f"\n  Clinical Note:")
    clinical_notes = {
        'LumA':   "Luminal A: ER+/PR+/HER2−, low Ki67. Best prognosis. Hormone therapy.",
        'LumB':   "Luminal B: ER+/PR+, can be HER2+, higher Ki67. Chemo + hormones.",
        'Her2':   "HER2-enriched: HER2 amplified. Trastuzumab + chemotherapy.",
        'Basal':  "Triple-Negative: ER−/PR−/HER2−. Chemotherapy. BRCA1/2 testing.",
        'Normal': "Normal-like: Good prognosis, similar to Luminal A management.",
    }
    lines.append(f"    {clinical_notes.get(pred_cls, '')}")
    lines.append(f"\n{'='*60}\n")

    return "\n".join(lines)


@torch.no_grad()
def predict(args, cfg, device):
    """Run single-patient inference."""

    # ── Load model ──────────────────────────────────────────────────────────
    model = HCMTClassifier.from_config(cfg)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model = model.to(device).eval()

    # ── Load modalities ──────────────────────────────────────────────────────
    n_clinical = cfg.data.n_clinical_features
    modalities_used = []

    wsi = None
    if args.wsi_features and Path(args.wsi_features).exists():
        wsi = torch.load(args.wsi_features, map_location='cpu').float()
        max_p = cfg.data.max_patches
        if wsi.shape[0] > max_p:
            wsi = wsi[:max_p]
        elif wsi.shape[0] < max_p:
            pad = torch.zeros(max_p - wsi.shape[0], wsi.shape[1])
            wsi = torch.cat([wsi, pad], dim=0)
        wsi = wsi.unsqueeze(0).to(device)  # (1, N, D)
        modalities_used.append('wsi')
        logger.info(f"  WSI features:  {wsi.shape}")

    genomics = None
    if args.genomics and Path(args.genomics).exists():
        genomics = torch.load(args.genomics, map_location='cpu').float()
        genomics = genomics.unsqueeze(0).to(device)  # (1, n_genes)
        modalities_used.append('genomics')
        logger.info(f"  Genomics:      {genomics.shape}")

    radiology = None
    if args.radiology and Path(args.radiology).exists():
        radiology = torch.load(args.radiology, map_location='cpu').float()
        if radiology.dim() == 3:
            radiology = radiology.unsqueeze(0)  # Add channel
        radiology = radiology.unsqueeze(0).to(device)  # (1, 1, D, H, W)
        modalities_used.append('radiology')
        logger.info(f"  Radiology:     {radiology.shape}")

    clinical = None
    if args.clinical_csv and Path(args.clinical_csv).exists():
        patient_id = args.patient_id or "patient"
        clinical = load_clinical_features(args.clinical_csv, patient_id, n_clinical)
        clinical = clinical.unsqueeze(0).to(device)  # (1, n_features)
        modalities_used.append('clinical')
        logger.info(f"  Clinical:      {clinical.shape}")

    if not modalities_used:
        raise ValueError("No valid modality files provided. Please supply at least one.")

    logger.info(f"  Modalities used: {modalities_used}")

    # ── Availability mask ────────────────────────────────────────────────────
    available = {
        'wsi':       torch.tensor([wsi       is not None]).to(device),
        'genomics':  torch.tensor([genomics  is not None]).to(device),
        'radiology': torch.tensor([radiology is not None]).to(device),
        'clinical':  torch.tensor([clinical  is not None]).to(device),
    }

    # ── Forward pass ─────────────────────────────────────────────────────────
    logits, gate_weights, _ = model(wsi, genomics, radiology, clinical, available)

    probs       = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    gate_w      = gate_weights.squeeze().cpu().numpy()

    # ── Print results ─────────────────────────────────────────────────────────
    pid = args.patient_id or "patient"
    report = format_prediction(pid, probs, gate_w, modalities_used)
    print(report)

    # ── Save JSON output ──────────────────────────────────────────────────────
    if args.output:
        import json
        result = {
            'patient_id':       pid,
            'predicted_subtype': PAM50_CLASSES[int(probs.argmax())],
            'confidence':        float(probs.max()),
            'probabilities':    {c: float(p) for c, p in zip(PAM50_CLASSES, probs)},
            'gate_weights':     {m: float(w) for m, w in
                                  zip(['wsi','genomics','radiology','clinical'], gate_w)},
            'modalities_used':  modalities_used
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to: {args.output}")

    return probs, gate_w


def main():
    parser = argparse.ArgumentParser(description="HCMT Single-Patient Inference")
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--config',       required=True)
    parser.add_argument('--patient_id',   default=None)
    parser.add_argument('--wsi_features', default=None, help='Path to .pt WSI features')
    parser.add_argument('--genomics',     default=None, help='Path to .pt genomics tensor')
    parser.add_argument('--radiology',    default=None, help='Path to .pt MRI volume')
    parser.add_argument('--clinical_csv', default=None, help='Path to clinical CSV')
    parser.add_argument('--output',       default=None, help='Optional path to save JSON result')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(args, cfg, device)


if __name__ == '__main__':
    main()
