# src/utils.py
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, path):
    """Save model + optimizer state to a single file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Restore from a checkpoint. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch']


def binary_iou(binary_logits, binary_mask, eps=1e-7):
    """IoU for the lane class. Lightweight metric for validation."""
    pred = binary_logits.argmax(dim=1)              # (B, H, W) in {0, 1}
    target = binary_mask.squeeze(1).long()           # (B, H, W) in {0, 1}
    intersection = ((pred == 1) & (target == 1)).sum().float()
    union        = ((pred == 1) | (target == 1)).sum().float()
    return (intersection / (union + eps)).item()