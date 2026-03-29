# HCMT — Hierarchical Cross-Modal Transformer
## Breast Cancer Subtype Classification via Multi-Modal Fusion

> Masters Research Project | Multi-Modal Medical AI | Computational Pathology

---

## Overview

HCMT fuses four clinical data modalities — Whole Slide Images (WSI), Genomics (RNA-seq),
Radiology (MRI/CT), and Clinical/EHR data — through a two-stage hierarchical attention
mechanism to classify breast cancer into PAM50 molecular subtypes.

**PAM50 Subtypes:**
- Luminal A (LumA)
- Luminal B (LumB)
- HER2-enriched (Her2)
- Triple-Negative / Basal-like (Basal)
- Normal-like (Normal)

---

## Architecture Summary

```
Raw Inputs (4 modalities)
        ↓
Modality-Specific Encoders
        ↓
Stage 1: Intra-Modal Self-Attention  (within each modality)
        ↓
Stage 2: Cross-Modal Attention        (each modality attends to all others)
        ↓
Gated Modality Fusion                 (learned per-sample modality weights)
        ↓
MLP Classifier  →  5 PAM50 Subtypes
```

---

## Project Structure

```
hcmt_project/
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── default.yaml           # Default hyperparameters
│   └── ablation.yaml          # Ablation study configs
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py        # Modality-specific encoders
│   │   ├── attention.py       # Intra- and cross-modal attention
│   │   ├── fusion.py          # Gated fusion module
│   │   └── hcmt.py            # Full HCMT model
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset for TCGA-BRCA
│   │   ├── preprocessing.py   # Data preprocessing pipelines
│   │   └── augmentation.py    # Data augmentation strategies
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Full training loop with logging
│   │   ├── losses.py          # Loss functions
│   │   └── scheduler.py       # LR schedulers
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # All evaluation metrics
│   │   └── evaluator.py       # Full evaluation pipeline
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── attention_viz.py   # Attention map visualization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Config loading utilities
│       ├── logging_utils.py   # Logging setup
│       └── checkpoint.py      # Model checkpoint utilities
│
├── scripts/
│   ├── preprocess_wsi.py      # WSI patch extraction via CLAM
│   ├── preprocess_genomics.py # RNA-seq normalization
│   ├── preprocess_radiology.py# MRI preprocessing
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Evaluation entry point
│   └── predict.py             # Single-sample inference
│
├── tests/
│   ├── test_encoders.py
│   ├── test_fusion.py
│   └── test_dataset.py
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_training_analysis.ipynb
    └── 03_attention_visualization.ipynb
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
# Extract WSI features (requires CLAM + CONCH installed)
python scripts/preprocess_wsi.py --wsi_dir /data/TCGA-BRCA/WSI --output_dir /data/features

# Normalize genomics
python scripts/preprocess_genomics.py --input /data/TCGA-BRCA/rnaseq.csv --output /data/genomics

# Preprocess MRI volumes
python scripts/preprocess_radiology.py --dicom_dir /data/TCIA --output_dir /data/radiology
```

### 3. Train
```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default.yaml
```

---

## Datasets

| Dataset     | Modalities            | N Patients | Access |
|-------------|-----------------------|-----------|--------|
| TCGA-BRCA   | WSI + RNA-seq + Clin  | 1,098     | GDC Portal (free, requires account) |
| TCIA Duke   | MRI                   | 922       | TCIA (free) |
| METABRIC    | Clinical + RNA-seq    | 2,509     | cBioPortal (free) |

---

## Citation

If you use this code, please cite:
```
@mastersthesis{hcmt2025,
  title  = {Hierarchical Cross-Modal Transformer Fusion for Breast Cancer Subtype Classification},
  author = {[Your Name]},
  year   = {2025},
  school = {[Your University]}
}
```
