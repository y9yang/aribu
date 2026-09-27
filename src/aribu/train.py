import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from .dataset import IGNORE_INDEX, LABEL_WATER, chip_input
from .metrics import confusion, metrics_from_counts
from .model import AMP, DEVICE

__all__ = ["BATCH", "ChipData", "build_cache", "masked_bce", "loss_iou", "train"]

BATCH = 8

class ChipData(Dataset):
    """Cached data (X, y), to be handed to a DataLoader."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

def build_cache(chip_ids, arm, mean, std, desc=""):
    """
    Build a cache of preprocessed chip data according to specified chip_ids, arm and normalization parameters. 
    
    Returns (X, y) as numpy arrays.
    """
    X = np.empty((len(chip_ids), 4 if arm == "s1+s2" else 2, 512, 512), np.float32)
    y = np.empty((len(chip_ids), 512, 512), np.uint8)
    for i, chip_id in enumerate(tqdm(chip_ids, desc=desc, leave=False)):
        X[i], y[i] = chip_input(chip_id, arm, mean, std)
    return X, y

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
def loss_iou(model, X, y, thresh=0.5):
    """Masked BCE and micro IoU over a cached split, the IoU counted by aribu.metrics.confusion.

    Returns (loss, iou), both pooled over every labelled pixel of the split.
    """
    model.eval()
    counts, total, n_pixels = np.zeros(4, np.int64), 0.0, 0          # counts: tp, fp, fn, tn
    for i in range(0, len(X), BATCH):
        with torch.autocast(**AMP):          # pyright: ignore[reportCallIssue, reportArgumentType]
            logits = model(torch.from_numpy(X[i:i + BATCH]).to(DEVICE)).float()
        yb = y[i:i + BATCH]
        n = int((yb != IGNORE_INDEX).sum())
        if n:                                # masked_bce is 0/0 on a batch with no labelled pixel
            total += masked_bce(logits, torch.from_numpy(yb).to(DEVICE)).item() * n
            n_pixels += n
        prob = torch.sigmoid(logits)[:, 0].cpu().numpy()
        counts += confusion(prob > thresh, yb == LABEL_WATER, yb != IGNORE_INDEX)
    return total / n_pixels, metrics_from_counts(counts)["iou"]

def train(model, loader, X_val, y_val, epochs=30, lr=3e-4):
    """Train a model with AdamW and mixed precision (AMP), keeping the best epoch by validation IoU.

    Returns the best val IoU and the per-epoch history (epoch, train_loss, val_loss, train_iou, val_iou).
    """
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    best_iou, best_state, history = -1.0, None, []

    bar = tqdm(range(1, epochs + 1))
    for epoch in bar:
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**AMP):          # pyright: ignore[reportCallIssue, reportArgumentType]
                loss = masked_bce(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        train_loss, train_iou = loss_iou(model, loader.dataset.X, loader.dataset.y)
        val_loss, val_iou = loss_iou(model, X_val, y_val)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_iou": train_iou, "val_iou": val_iou})
        if val_iou > best_iou:
            best_iou, best_state = val_iou, copy.deepcopy(model.state_dict())
        bar.set_postfix(train_iou=f"{train_iou:.4f}", val_iou=f"{val_iou:.4f}")

    model.load_state_dict(best_state)
    return best_iou, pd.DataFrame(history)
