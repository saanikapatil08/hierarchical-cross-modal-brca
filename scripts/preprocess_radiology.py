"""
scripts/preprocess_radiology.py
================================
MRI/CT Volume Preprocessing Pipeline for HCMT.

Uses the TCIA Duke Breast Cancer MRI dataset:
    https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/

Steps:
    1. Load DICOM series for each patient
    2. Convert to NIfTI / numpy volume
    3. Resample to isotropic resolution (1mm³)
    4. Normalize intensity (z-score or percentile clipping)
    5. Crop to breast region using bounding box
    6. Resize to target shape (D × H × W)
    7. Save as .pt tensors

Requirements:
    pip install SimpleITK pydicom nibabel

Usage:
    python scripts/preprocess_radiology.py \
        --dicom_dir /data/TCIA/Duke-Breast-Cancer-MRI \
        --output_dir /data/tcga_brca/radiology \
        --target_size 32 128 128 \
        --modality DCE

Patient ID matching:
    TCIA patient IDs follow the format: Breast_MRI_001, Breast_MRI_002, ...
    You will need a clinical mapping CSV to link TCIA IDs to TCGA patient IDs.
    This script saves files using TCIA IDs; match them to TCGA IDs separately.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_dicom_series(patient_dir: str) -> tuple:
    """
    Load a DICOM series from a directory using SimpleITK.

    Handles:
        - Multi-frame DICOM
        - Series with multiple time points (DCE-MRI)
        - Automatic series UID detection

    Args:
        patient_dir (str): Directory containing DICOM files (.dcm).

    Returns:
        Tuple (volume: np.ndarray, spacing: tuple) where
        volume is (D, H, W) and spacing is (dz, dy, dx) in mm.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        print("Install SimpleITK: pip install SimpleITK")
        raise

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(patient_dir)

    if not series_ids:
        raise ValueError(f"No DICOM series found in: {patient_dir}")

    # For DCE-MRI, take the first post-contrast series (typically longest series)
    # In practice you may need to filter by SeriesDescription
    best_series = max(series_ids, key=lambda sid: len(
        reader.GetGDCMSeriesFileNames(patient_dir, sid)
    ))

    dicom_files = reader.GetGDCMSeriesFileNames(patient_dir, best_series)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    volume  = sitk.GetArrayFromImage(image)   # (D, H, W) in numpy
    spacing = image.GetSpacing()              # (x_spacing, y_spacing, z_spacing)

    return volume.astype(np.float32), spacing


def resample_volume(
    volume: np.ndarray,
    original_spacing: tuple,
    target_spacing: tuple = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Resample volume to isotropic spacing using trilinear interpolation.

    Why resample?
        DICOM volumes can have non-isotropic voxels (e.g., 0.7 × 0.7 × 3.0 mm).
        Resampling ensures consistent spatial resolution across patients.

    Args:
        volume           (ndarray): Input volume (D, H, W).
        original_spacing (tuple):   Original voxel spacing (x, y, z) in mm.
        target_spacing   (tuple):   Target voxel spacing in mm.

    Returns:
        Resampled volume (ndarray).
    """
    import torch.nn.functional as F

    orig = np.array(original_spacing)[[2, 1, 0]]  # Convert to (z, y, x)
    tgt  = np.array(target_spacing)

    new_shape = np.round(
        np.array(volume.shape) * orig / tgt
    ).astype(int)

    vol_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    resampled  = F.interpolate(
        vol_tensor,
        size=tuple(new_shape),
        mode='trilinear',
        align_corners=False
    )
    return resampled.squeeze().numpy()


def normalize_intensity(
    volume: np.ndarray,
    method: str = 'percentile'
) -> np.ndarray:
    """
    Normalize MRI intensity values.

    Methods:
        'zscore'     — Zero mean, unit variance (global)
        'percentile' — Clip to 1st–99th percentile, then scale to [0, 1]
        'minmax'     — Scale to [0, 1] based on min/max

    Args:
        volume (ndarray): Input volume.
        method (str):     Normalization method.

    Returns:
        Normalized volume (ndarray, float32).
    """
    if method == 'zscore':
        mean = volume.mean()
        std  = volume.std() + 1e-8
        return ((volume - mean) / std).astype(np.float32)

    elif method == 'percentile':
        p1, p99 = np.percentile(volume, [1, 99])
        volume  = np.clip(volume, p1, p99)
        volume  = (volume - p1) / (p99 - p1 + 1e-8)
        return volume.astype(np.float32)

    elif method == 'minmax':
        vmin, vmax = volume.min(), volume.max()
        return ((volume - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_volume(volume: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize volume to exact target spatial dimensions.

    Args:
        volume      (ndarray): (D, H, W) volume.
        target_size (tuple):   Target (D, H, W).

    Returns:
        Resized volume (ndarray).
    """
    import torch
    import torch.nn.functional as F

    vol_t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(
        vol_t,
        size=target_size,
        mode='trilinear',
        align_corners=False
    )
    return resized.squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description="MRI Volume Preprocessing for HCMT")
    parser.add_argument('--dicom_dir',  required=True, help='Root dir with per-patient DICOM subdirs')
    parser.add_argument('--output_dir', required=True, help='Output dir for .pt tensors')
    parser.add_argument('--target_size', nargs=3, type=int, default=[32, 128, 128],
                        help='Target (D H W)')
    parser.add_argument('--normalization', default='percentile',
                        choices=['zscore', 'percentile', 'minmax'])
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        help='Target isotropic spacing in mm')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.target_size)

    # Find patient directories
    patient_dirs = sorted([
        d for d in Path(args.dicom_dir).iterdir() if d.is_dir()
    ])
    print(f"Found {len(patient_dirs)} patient directories")

    for patient_dir in tqdm(patient_dirs, desc="Preprocessing MRI"):
        patient_id = patient_dir.name
        out_path   = output_dir / f"{patient_id}.pt"

        if out_path.exists():
            continue

        try:
            # Load DICOM
            volume, spacing = load_dicom_series(str(patient_dir))

            # Resample to isotropic
            volume = resample_volume(volume, spacing, tuple(args.spacing))

            # Normalize intensity
            volume = normalize_intensity(volume, method=args.normalization)

            # Resize to target
            volume = resize_volume(volume, target_size)

            # Add channel dim: (D, H, W) → (1, D, H, W)
            tensor = torch.from_numpy(volume).unsqueeze(0)

            torch.save(tensor, out_path)
            tqdm.write(f"  {patient_id}: {tensor.shape}")

        except Exception as e:
            tqdm.write(f"ERROR: {patient_id}: {e}")

    print(f"Done. Output: {output_dir}")


if __name__ == '__main__':
    main()
