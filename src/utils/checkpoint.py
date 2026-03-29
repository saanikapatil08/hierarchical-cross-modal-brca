import torch
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})
