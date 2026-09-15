import os
import numpy as np
# pandas must load before torchvision (pulled in by smp): the reverse order crashes Python on Windows (0xC0000374)
import pandas  # noqa: F401
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
from .dataset import LABEL_WATER, load_chip, occluded_input, valid_mask
from .metrics import confusion, metrics_from_counts

__all__ = ["DEVICE", "AMP", "ENCODER", "build_unet", "load_checkpoint", "prob_from_input", "chip_prob",
           "make_predictor", "micro_iou"]

# ARIBU_DEVICE=cpu forces the CPU, so live runs on the laptop take as long as on a CPU-only host
DEVICE = os.environ.get("ARIBU_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
AMP = dict(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda")
ENCODER = "resnet34"

def build_unet(in_channels):
    """Build an `smp.Unet` with ImageNet-pretrained ResNet-34 encoder."""
    return smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=in_channels, classes=1)

def load_checkpoint(path):
    """Rebuild a frozen U-Net in eval mode on DEVICE.

    Returns (model, checkpoint); the checkpoint holds arm, threshold and normalisation.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = smp.Unet(ENCODER, encoder_weights=None, in_channels=ckpt["in_channels"], classes=1)
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return model.to(DEVICE).eval(), ckpt

@torch.inference_mode()
def prob_from_input(model, x):
    """Water probability (H, W) from a model input `x`, a NumPy array (C, H, W)."""
    xb = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(DEVICE)
    with torch.autocast(**AMP):          # pyright: ignore[reportCallIssue, reportArgumentType]
        return torch.sigmoid(model(xb).float())[0, 0].cpu().numpy()

def chip_prob(model, arm, chip_id, mean, std, fraction=0.0):
    """
    Water probability (H, W) for one chip.

    Optional: `fraction` the of chip covered by synthetic clouds.
    """
    return prob_from_input(model, occluded_input(chip_id, arm, fraction, mean, std)[0])

def make_predictor(model, arm, mean, std, thresh=0.5):
    """Return the trained model for the specified arm, preprocessing statistics, and decision threshold."""
    model = model.to(DEVICE).eval()

    def for_chip(chip_id):
        prob = chip_prob(model, arm, chip_id, mean, std)
        return lambda vv, vh, valid: (prob > thresh) & valid

    return for_chip

def micro_iou(model, arm, chip_ids, thresholds, mean, std, fraction=0.0, desc=""):
    """
    Given a list of thresholds, score the model for the given arm by micro IoU for each threshold over a list of chips.

    Optional: `fraction` the of chip covered by synthetic clouds.
    """
    model.to(DEVICE).eval()
    counts = np.zeros((len(thresholds), 4), np.int64)
    for chip_id in tqdm(chip_ids, desc=desc, leave=False):
        c = load_chip(chip_id)
        v = valid_mask(c)
        if not v.any():
            continue
        p = chip_prob(model, arm, chip_id, mean, std, fraction)
        w = c.label == LABEL_WATER
        for j, t in enumerate(thresholds):
            counts[j] += confusion(p > t, w, v)
    return np.array([metrics_from_counts(row)["iou"] for row in counts])
