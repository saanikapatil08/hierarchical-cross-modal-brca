# HCMT — Hierarchical Cross-Modal Transformer
## Breast Cancer PAM50 Subtype Classification via Multi-Modal Fusion

> Research Project | Multi-Modal Medical AI | TCGA-BRCA | PyTorch

---

## Results (Genomics + Clinical, No WSI/Radiology)

| Metric | Validation | Test |
|--------|-----------|------|
| Macro F1 | 0.813 | 0.677 |
| Balanced Accuracy | 0.789 | 0.640 |
| Mean AUC | — | 0.899 |
| Cohen's Kappa | — | 0.664 |

**Per-class test performance:**

| Subtype | Precision | Recall | F1 | AUC |
|---------|-----------|--------|----|-----|
| Luminal A | 0.803 | 0.867 | 0.833 | 0.919 |
| Luminal B | 0.588 | 0.667 | 0.625 | 0.894 |
| HER2-enriched | 0.875 | 0.583 | 0.700 | 0.798 |
| Basal-like | 1.000 | 0.885 | **0.939** | 0.925 |
| Normal-like | 0.500 | 0.200 | 0.286 | 0.958 |

> Best model checkpoint: epoch 26 of 41 (early stopping, patience=15)

---

## Overview

HCMT fuses clinical data modalities through a two-stage hierarchical attention mechanism to classify breast cancer into PAM50 molecular subtypes. Currently trained on **genomics (RNA-seq) + clinical** data from TCGA-BRCA. WSI and radiology modalities are implemented but not yet trained.

**PAM50 Subtypes:**
- Luminal A (LumA) — 499 patients (50.9%)
- Luminal B (LumB) — 197 patients (20.1%)
- Basal-like (Basal) — 171 patients (17.4%)
- HER2-enriched (Her2) — 78 patients (8.0%)
- Normal-like (Normal) — 36 patients (3.7%)

---

## Architecture

```
Inputs: Genomics (60,660 genes) + Clinical (17 features)
        ↓
Modality-Specific Encoders
  - Genomics: Linear projection → 64 tokens × 256 dim  (39.5M params)
  - Clinical: Linear projection → 16 tokens × 256 dim  (0.5M params)
        ↓
Stage 1: Intra-Modal Self-Attention  (2 layers, 8 heads per modality)
        ↓
Stage 2: Cross-Modal Attention       (genomics ↔ clinical, 2 layers)
        ↓
Gated Modality Fusion                (learned per-sample modality weights)
        ↓
MLP Classifier → 5 PAM50 Subtypes

Total parameters: 54,184,717
```

---

## Dataset

| Statistic | Value |
|-----------|-------|
| Source | TCGA-BRCA via GDC Data Portal |
| Total patients | 981 (clinical + genomics overlap) |
| Genomics | 60,660 genes, log1p(TPM) normalized |
| Clinical features | 17 (after one-hot encoding) |
| Train / Val / Test | 686 / 147 / 148 |
| GPU | NVIDIA Tesla T4 (Google Colab) |
| Training time | ~7 minutes (41 epochs) |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Download TCGA-BRCA RNA-seq data from [GDC Portal](https://portal.gdc.cancer.gov/):
```bash
# Normalize genomics (produces .pt tensors)
python scripts/preprocess_genomics.py \
  --input_dir data/raw/rnaseq/ \
  --output_dir data/tcga_brca/genomics/

# Prepare clinical CSV
python scripts/prepare_clinical.py
```

The processed data zip (`tcga_brca_processed.zip`) is available in Google Drive (see Artifacts section).

### 3. Train
```bash
WANDB_MODE=disabled python scripts/train.py \
  --config configs/default.yaml \
  data.use_wsi=false \
  data.use_radiology=false \
  data.label_column=PAM50 \
  experiment.name=genomics_clinical_baseline
```

### 4. Evaluate
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pth \
  --config configs/default.yaml
```

---

## Project Structure

```
hierarchical-cross-modal-brca/
├── configs/
│   ├── default.yaml           # Training config (n_genes=60660, n_clinical=17)
│   └── ablation.yaml          # Ablation study configs
├── src/
│   ├── models/
│   │   ├── hcmt.py            # HCMTClassifier (main model)
│   │   ├── encoders.py        # Genomics, clinical, WSI, radiology encoders
│   │   ├── attention.py       # Intra- and cross-modal attention
│   │   └── fusion.py          # Gated fusion module
│   ├── data/
│   │   └── dataset.py         # TCGABRCADataset
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Cross-entropy with label smoothing
│   │   └── scheduler.py       # Cosine warmup scheduler
│   ├── evaluation/
│   │   └── metrics.py         # F1, AUC, kappa, confusion matrix
│   └── utils/
│       ├── checkpoint.py      # save/load checkpoints
│       └── logging_utils.py   # Logger setup
├── scripts/
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation entry point
│   ├── preprocess_genomics.py # RNA-seq → .pt tensors
│   └── prepare_clinical.py    # Clinical CSV preparation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_gpu_training_colab.ipynb   # Full Colab training notebook
│   └── 03_attention_visualization.ipynb
├── tests/
├── requirements.txt
└── HCMT_Project_Documentation.docx  # Full project report
```

---

## Key Config Values

```yaml
data:
  n_genes: 60660           # GENCODE v36 gene count
  n_clinical_features: 17  # After one-hot encoding
  label_column: PAM50

training:
  epochs: 100
  early_stopping_patience: 15
  lr: 0.0003
  scheduler: cosine_warmup
  warmup_epochs: 10
  use_amp: false           # Disabled — gradient overflow with T4
  use_class_weights: true  # Handles class imbalance
```

---

## Stored Artifacts

| Artifact | Location |
|----------|----------|
| Best checkpoint (epoch 26) | Google Drive: `hcmt_best_epoch26_f1_0.8127.pth` |
| Processed data (981 patients) | Google Drive: `tcga_brca_processed.zip` (151MB) |
| Full project report | `HCMT_Project_Documentation.docx` |

---

## Next Steps

- [ ] Add WSI modality (patch feature extraction with UNI/CONCH)
- [ ] Ablation studies: genomics-only vs clinical-only vs combined
- [ ] 5-fold cross-validation
- [ ] Radiology (MRI) integration
- [ ] Attention visualization for interpretability
- [ ] External validation on METABRIC

---

