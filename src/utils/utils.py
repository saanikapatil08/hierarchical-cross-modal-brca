"""
src/utils/checkpoint.py   — save/load model checkpoints
src/utils/logging_utils.py — logger setup
src/utils/config.py       — config loading helpers
"""

# ─── checkpoint.py ────────────────────────────────────────────────────────────
import torch
import logging
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str
):
    """
    Save model checkpoint including optimizer state and metrics.

    Args:
        model     : The model to save.
        optimizer : Optimizer (saved for potential resuming).
        epoch     : Current epoch number.
        metrics   : Validation metrics dict.
        path      : File path (.pth).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics':         metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load a saved checkpoint into a model.

    Args:
        model     : Model to load weights into.
        path      : Path to .pth checkpoint file.
        optimizer : Optional — if provided, loads optimizer state too.
        device    : Device to load tensors to.

    Returns:
        Dict with 'epoch' and 'metrics' keys.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return {'epoch': ckpt.get('epoch', 0), 'metrics': ckpt.get('metrics', {})}


# ─── logging_utils.py ─────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a named logger with consistent formatting.

    Args:
        name  : Logger name (use __name__ in calling module).
        level : Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            '[%(asctime)s] %(levelname)s %(name)s — %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── config.py ────────────────────────────────────────────────────────────────

def load_config(config_path: str):
    """
    Load OmegaConf config from a YAML file.

    Args:
        config_path (str): Path to .yaml config file.

    Returns:
        OmegaConf DictConfig.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    return cfg


def set_seed(seed: int):
    """
    Set random seeds for full reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also enables deterministic CuDNN (may reduce speed slightly).

    Args:
        seed (int): Random seed value.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
