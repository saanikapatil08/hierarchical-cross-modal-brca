"""
scripts/preprocess_genomics.py
==============================
RNA-seq Genomics Preprocessing Pipeline for TCGA-BRCA.

Steps:
    1. Load raw TCGA-BRCA RNA-seq data (HTSeq counts or FPKM)
    2. Filter low-variance / low-expression genes
    3. Log1p normalization (standard for RNA-seq ML)
    4. Optional: variance-based gene selection (top-k most variable)
    5. Optional: z-score normalization across patients
    6. Save per-patient tensors as .pt files

Input:
    TCGA-BRCA RNA-seq can be downloaded from:
        https://portal.gdc.cancer.gov/  (GDC Portal)
    Or via TCGABiolinks in R / Python:
        from tcga_biolinks import ... (see below)

    Expected CSV format (gene rows × patient columns):
        gene_id,TCGA-A1-A0SK,TCGA-A1-A0SP,...
        ENSG00000000003,145.2,230.1,...

Usage:
    python scripts/preprocess_genomics.py \
        --input /data/TCGA-BRCA/rnaseq_fpkm.csv \
        --output_dir /data/tcga_brca/genomics \
        --n_top_genes 20531 \
        --normalization log1p_zscore
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_tcga_rnaseq(input_path: str) -> pd.DataFrame:
    """
    Load TCGA RNA-seq matrix.

    The GDC portal provides gene expression as a matrix where:
        - Rows = genes (Ensembl IDs or gene symbols)
        - Columns = TCGA sample barcodes (TCGA-XX-XXXX-01A-...)

    We normalize sample IDs to patient IDs (first 12 chars).

    Returns:
        pd.DataFrame with patient IDs as columns, gene IDs as index.
    """
    print(f"Loading RNA-seq data from: {input_path}")
    df = pd.read_csv(input_path, index_col=0)

    # Normalize column names: TCGA-A1-A0SK-01A-11R-... → TCGA-A1-A0SK
    df.columns = ['-'.join(c.split('-')[:3]) for c in df.columns]

    # Remove duplicate samples (keep first)
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"  Loaded: {df.shape[0]} genes × {df.shape[1]} patients")
    return df


def filter_genes(
    df: pd.DataFrame,
    min_expression: float = 1.0,
    min_patients_expressed: float = 0.1
) -> pd.DataFrame:
    """
    Remove genes with very low or no expression.

    Criteria:
        - At least 10% of patients must have expression > min_expression
        - Removes unexpressed/artifact genes

    Args:
        df                    : Gene × Patient DataFrame.
        min_expression        : Minimum expression threshold.
        min_patients_expressed: Fraction of patients that must express the gene.

    Returns:
        Filtered DataFrame.
    """
    n_patients = df.shape[1]
    threshold  = int(n_patients * min_patients_expressed)

    expressed_mask = (df > min_expression).sum(axis=1) >= threshold
    df_filtered = df[expressed_mask]

    print(f"  After filtering: {df_filtered.shape[0]} genes retained "
          f"(removed {df.shape[0] - df_filtered.shape[0]})")
    return df_filtered


def select_top_variable_genes(df: pd.DataFrame, n_top: int = 20531) -> pd.DataFrame:
    """
    Select the top-k most variable genes by variance (MAD or std).

    Variable genes carry the most biological information for subtype discrimination.

    Args:
        df    : Gene × Patient DataFrame (already log-normalized).
        n_top : Number of genes to keep.

    Returns:
        DataFrame with only top-k genes.
    """
    if df.shape[0] <= n_top:
        return df

    # Use Median Absolute Deviation (robust to outliers)
    mad = df.subtract(df.median(axis=1), axis=0).abs().median(axis=1)
    top_genes = mad.nlargest(n_top).index
    return df.loc[top_genes]


def normalize_rnaseq(
    df: pd.DataFrame,
    method: str = 'log1p_zscore'
) -> pd.DataFrame:
    """
    Normalize RNA-seq data.

    Methods:
        'log1p'         — log(x + 1) transformation only
        'log1p_zscore'  — log1p then z-score across patients (recommended)
        'tpm'           — Normalize to TPM if providing raw counts (not implemented here)

    Args:
        df     : Gene × Patient DataFrame.
        method : Normalization method.

    Returns:
        Normalized DataFrame (same shape).
    """
    if 'log1p' in method:
        df = np.log1p(df)

    if 'zscore' in method:
        # Z-score per gene across patients (zero mean, unit variance)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.T).T  # Fit on patients, scale genes
        df = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)
        print("  Applied log1p + z-score normalization")
    else:
        print("  Applied log1p normalization")

    return df


def main():
    parser = argparse.ArgumentParser(description="TCGA-BRCA RNA-seq Preprocessing")
    parser.add_argument('--input',       required=True, help='Path to RNA-seq CSV (genes × patients)')
    parser.add_argument('--output_dir',  required=True, help='Output directory for .pt tensors')
    parser.add_argument('--n_top_genes', type=int, default=20531,
                        help='Number of top variable genes to keep')
    parser.add_argument('--normalization', default='log1p_zscore',
                        choices=['log1p', 'log1p_zscore'],
                        help='Normalization method')
    parser.add_argument('--min_expression', type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    df = load_tcga_rnaseq(args.input)

    # ── Filter ─────────────────────────────────────────────────────────────────
    df = filter_genes(df, min_expression=args.min_expression)

    # ── Normalize ──────────────────────────────────────────────────────────────
    df = normalize_rnaseq(df, method=args.normalization)

    # ── Select top variable genes ──────────────────────────────────────────────
    df = select_top_variable_genes(df, n_top=args.n_top_genes)
    print(f"Final gene matrix: {df.shape[0]} genes × {df.shape[1]} patients")

    # Save gene list for reference
    gene_list_path = output_dir / 'gene_list.txt'
    df.index.to_series().to_csv(gene_list_path, index=False, header=False)
    print(f"Gene list saved to: {gene_list_path}")

    # ── Save per-patient tensors ────────────────────────────────────────────────
    print("Saving per-patient tensors...")
    for patient_id in tqdm(df.columns):
        expr = df[patient_id].values.astype(np.float32)
        tensor = torch.tensor(expr)
        torch.save(tensor, output_dir / f"{patient_id}.pt")

    print(f"Done. Saved {len(df.columns)} patient tensors to: {output_dir}")
    print(f"Tensor shape per patient: ({df.shape[0]},)")


if __name__ == '__main__':
    main()
