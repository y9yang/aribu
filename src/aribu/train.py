import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from .dataset import IGNORE_INDEX, LABEL_WATER, chip_input
from .metrics import confusion, metrics_from_counts
from .model import AMP, DEVICE

__all__ = ["BATCH", "ChipData", "build_cache", "masked_bce", "val_iou", "train"]

BATCH = 8

class ChipData(Dataset):
    """Cached data (xs, ys), to be handed to a DataLoader."""

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return torch.from_numpy(self.xs[i]), torch.from_numpy(self.ys[i])

def build_cache(chip_ids, arm, mean, std, desc=""):
    """
    Build a cache of preprocessed chip data according to specified chip_ids, arm and normalization parameters. 
    
    Returns (xs, ys) as numpy arrays.
    """
    xs = np.empty((len(chip_ids), 4 if arm == "s1+s2" else 2, 512, 512), np.float32)
    ys = np.empty((len(chip_ids), 512, 512), np.uint8)
    for i, chip_id in enumerate(tqdm(chip_ids, desc=desc, leave=False)):
        xs[i], ys[i] = chip_input(chip_id, arm, mean, std)
    return xs, ys

def masked_bce(logits, y):
    """
    BCE over labelled pixels only.
    
    logits (B, 1, H, W), y (B, H, W) in {LABEL_LAND, LABEL_WATER, IGNORE_INDEX}.
    """
    keep = (y != IGNORE_INDEX).float()
    target = (y == LABEL_WATER).float()
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), target, reduction="none")  # no reduction --> pixel-wise BCE
    return (loss * keep).sum() / keep.sum()         # the pixel-wise multiplication kills the loss for invalid pixels

@torch.no_grad()
def val_iou(model, xs, ys, thresh=0.5):
    """Micro IoU over a cached split, counted by aribu.metrics.confusion."""
    model.eval()
    counts = np.zeros(4, np.int64)          # tp, fp, fn, tn
    for i in range(0, len(xs), BATCH):
        with torch.autocast(**AMP):          # pyright: ignore[reportCallIssue, reportArgumentType]
            logits = model(torch.from_numpy(xs[i:i + BATCH]).to(DEVICE))
        prob = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        y = ys[i:i + BATCH]
        counts += confusion(prob > thresh, y == LABEL_WATER, y != IGNORE_INDEX)
    return metrics_from_counts(counts)["iou"]

def train(model, loader, xva, yva, epochs=30, lr=3e-4):
    """Train a model with AdamW and mixed precision (AMP), keeping the best epoch by validation IoU.

    Returns the best IoU and per-epoch history.
    """
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    best_iou, best_state, history = -1.0, None, []

    bar = tqdm(range(1, epochs + 1))
    for epoch in bar:
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**AMP):          # pyright: ignore[reportCallIssue, reportArgumentType]
                loss = masked_bce(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()

        loss, iou = total / len(loader), val_iou(model, xva, yva)
        history.append({"epoch": epoch, "loss": loss, "val_iou": iou})
        if iou > best_iou:
            best_iou, best_state = iou, copy.deepcopy(model.state_dict())
        bar.set_postfix(loss=f"{loss:.4f}", val_iou=f"{iou:.4f}")

    model.load_state_dict(best_state)
    return best_iou, pd.DataFrame(history)
