"""
scripts/preprocess_wsi.py
=========================
WSI Patch Extraction and Feature Encoding Pipeline.

Overview:
    1. Read WSI files (.svs/.ndpi/.tiff) from TCGA-BRCA download
    2. Segment tissue using CLAM's tissue segmentation algorithm
    3. Extract non-overlapping 256×256 patches at 20× magnification
    4. Encode each patch with a pathology foundation model (CONCH/UNI)
    5. Save per-patient feature tensors as .pt files

Requirements:
    pip install openslide-python
    pip install git+https://github.com/mahmoodlab/CLAM.git
    # For CONCH: follow https://huggingface.co/MahmoodLab/CONCH
    # For UNI:   follow https://huggingface.co/MahmoodLab/UNI

Usage:
    python scripts/preprocess_wsi.py \
        --wsi_dir /data/TCGA-BRCA/WSI \
        --output_dir /data/tcga_brca/wsi_features \
        --encoder conch \
        --batch_size 256 \
        --device cuda

Expected input structure:
    wsi_dir/
        TCGA-A1-A0SK-01Z-00-DX1.svs
        TCGA-A1-A0SP-01Z-00-DX1.svs
        ...

Output structure:
    output_dir/
        TCGA-A1-A0SK.pt    # (N_patches, 512) for CONCH, (N_patches, 1024) for UNI
        TCGA-A1-A0SP.pt
        ...
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_patient_id(wsi_path: str) -> str:
    """
    Extract TCGA patient ID from WSI filename.
    e.g., 'TCGA-A1-A0SK-01Z-00-DX1.svs' → 'TCGA-A1-A0SK'
    """
    stem = Path(wsi_path).stem
    parts = stem.split('-')
    return '-'.join(parts[:3])  # TCGA-XX-XXXX


def load_encoder(encoder_name: str, device: str):
    """
    Load a pathology foundation model encoder.

    Supported:
        'conch'   — CONCH (MahmoodLab/CONCH on HuggingFace), outputs 512-dim
        'uni'     — UNI   (MahmoodLab/UNI on HuggingFace), outputs 1024-dim
        'plip'    — PLIP  (vinid5/plip on HuggingFace), outputs 512-dim
        'resnet50'— Standard ResNet-50 pre-trained on ImageNet (baseline)

    For CONCH/UNI: requires HuggingFace access token
        export HF_TOKEN=<your_token>

    Returns:
        (model, transform, feature_dim)
    """
    import torchvision.transforms as T

    base_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if encoder_name == 'conch':
        # CONCH: vision-language model for pathology
        # Paper: https://www.nature.com/articles/s41591-024-02856-4
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            model, transform = create_model_from_pretrained(
                'conch_ViT-B-16', "hf_hub:MahmoodLab/conch"
            )
            model = model.visual  # Vision encoder only
            feat_dim = 512
        except ImportError:
            print("CONCH not installed. Install via: pip install git+https://github.com/mahmoodlab/CONCH")
            raise

    elif encoder_name == 'uni':
        # UNI: large pathology foundation model
        # Paper: https://www.nature.com/articles/s41591-024-02857-3
        try:
            import timm
            model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True, init_values=1e-5, dynamic_img_size=True
            )
            feat_dim = 1024
            transform = base_transform
        except Exception:
            print("UNI requires timm and HuggingFace access token.")
            raise

    elif encoder_name == 'resnet50':
        # Baseline: standard ResNet-50 (no domain-specific pretraining)
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        feat_dim = 2048
        transform = base_transform

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose: conch, uni, resnet50")

    model = model.to(device).eval()
    return model, transform, feat_dim


def extract_patches_clam(wsi_path: str, patch_size: int = 256, magnification: int = 20):
    """
    Use CLAM's segmentation to extract tissue patches from a WSI.

    Steps:
        1. Downsample WSI to thumbnail for tissue/background segmentation
        2. Otsu thresholding to create tissue mask
        3. Extract coordinates of non-overlapping patches in tissue regions

    Args:
        wsi_path     (str): Path to .svs file.
        patch_size   (int): Patch size in pixels.
        magnification(int): Target magnification (20 or 40×).

    Returns:
        List of PIL Images (patch_size × patch_size).
    """
    try:
        import openslide
        from PIL import Image
        import cv2
    except ImportError:
        print("Install: pip install openslide-python pillow opencv-python")
        raise

    slide = openslide.OpenSlide(wsi_path)

    # Get target level for requested magnification
    obj_power = float(slide.properties.get('openslide.objective-power', 40))
    target_level = slide.get_best_level_for_downsample(obj_power / magnification)

    # Thumbnail for segmentation
    thumb_size = (1024, 1024)
    thumbnail = slide.get_thumbnail(thumb_size)
    thumb_arr = np.array(thumbnail.convert('RGB'))

    # Tissue segmentation: Otsu on green channel
    gray = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2GRAY)
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

    # Get slide dimensions at target level
    level_dims = slide.level_dimensions[target_level]
    W, H = level_dims

    # Sample patch coordinates from tissue regions
    scale_x = W / thumb_size[0]
    scale_y = H / thumb_size[1]

    patches = []
    patch_stride = patch_size  # No overlap

    for y in range(0, H - patch_size, patch_stride):
        for x in range(0, W - patch_size, patch_stride):
            # Check if center of patch is in tissue
            tx = int(x / scale_x)
            ty = int(y / scale_y)
            if tissue_mask[ty, tx] > 0:
                # Extract patch from slide at target level
                region = slide.read_region(
                    location=(int(x * slide.level_downsamples[target_level]),
                               int(y * slide.level_downsamples[target_level])),
                    level=target_level,
                    size=(patch_size, patch_size)
                )
                patches.append(region.convert('RGB'))

    slide.close()
    return patches


@torch.no_grad()
def encode_patches(
    patches,
    model,
    transform,
    batch_size: int = 256,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Encode a list of PIL patches using the encoder model.

    Args:
        patches    (list): List of PIL Images.
        model      : Encoder model.
        transform  : Preprocessing transform.
        batch_size (int): Encoding batch size.
        device     (str): Device string.

    Returns:
        torch.Tensor: (N, feat_dim) feature matrix.
    """
    all_feats = []

    for i in range(0, len(patches), batch_size):
        batch = patches[i: i + batch_size]
        tensors = torch.stack([transform(p) for p in batch]).to(device)
        feats = model(tensors)

        # Handle different output shapes
        if feats.dim() == 4:             # (B, C, H, W) — spatial output
            feats = feats.mean(dim=[2, 3])  # Global average pool
        elif feats.dim() == 3:           # (B, T, C) — ViT tokens
            feats = feats[:, 0, :]          # Use CLS token
        elif feats.dim() == 2:           # (B, C) — already flat
            pass

        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)  # (N, feat_dim)


def main():
    parser = argparse.ArgumentParser(description="WSI Feature Extraction for HCMT")
    parser.add_argument('--wsi_dir',    required=True, help='Directory containing .svs files')
    parser.add_argument('--output_dir', required=True, help='Output directory for .pt feature files')
    parser.add_argument('--encoder',    default='conch', choices=['conch', 'uni', 'resnet50'],
                        help='Feature encoder model')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_patches',type=int, default=4096)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"Loading encoder: {args.encoder}")
    model, transform, feat_dim = load_encoder(args.encoder, args.device)
    print(f"Encoder loaded. Feature dim: {feat_dim}")

    # Find WSI files
    wsi_files = list(Path(args.wsi_dir).glob('*.svs')) + \
                list(Path(args.wsi_dir).glob('*.ndpi')) + \
                list(Path(args.wsi_dir).glob('*.tiff'))
    print(f"Found {len(wsi_files)} WSI files")

    for wsi_path in tqdm(wsi_files, desc="Processing WSIs"):
        patient_id = get_patient_id(str(wsi_path))
        out_path   = output_dir / f"{patient_id}.pt"

        if out_path.exists():
            continue  # Skip already processed

        try:
            # Extract patches
            patches = extract_patches_clam(str(wsi_path))
            if len(patches) == 0:
                print(f"WARNING: No tissue patches found for {patient_id}")
                continue

            # Limit to max_patches (random sample)
            if len(patches) > args.max_patches:
                idx     = np.random.choice(len(patches), args.max_patches, replace=False)
                patches = [patches[i] for i in idx]

            # Encode
            feats = encode_patches(patches, model, transform, args.batch_size, args.device)

            # Save
            torch.save(feats, out_path)
            tqdm.write(f"  {patient_id}: {feats.shape}")

        except Exception as e:
            tqdm.write(f"ERROR processing {patient_id}: {e}")


if __name__ == '__main__':
    main()
