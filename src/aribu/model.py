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

# ARIBU_DEVICE="cpu" forces the CPU, so live page will always run on CPU
DEVICE = os.environ.get("ARIBU_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
AMP = {"device_type": DEVICE, "enabled": DEVICE == "cuda"}
ENCODER = "resnet34"

def build_unet(in_channels, encoder_weights="imagenet"):
    """Build an `smp.Unet` with a ResNet-34 encoder, ImageNet-pretrained by default."""
    return smp.Unet(ENCODER, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)

def load_checkpoint(path):
    """Rebuild a frozen U-Net in eval mode on DEVICE.

    Returns (model, checkpoint); the checkpoint holds arm, threshold and normalisation.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_unet(ckpt["in_channels"], encoder_weights=None)   # pyright: ignore[reportArgumentType]
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return model.to(DEVICE).eval(), ckpt

@torch.inference_mode()
def prob_from_input(model, x):
    """Water probability (H, W) from a model input `x`, a NumPy array (C, H, W)."""
    xb = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    with torch.autocast(**AMP):          # pyright: ignore[reportCallIssue, reportArgumentType]
        return torch.sigmoid(model(xb).float()).squeeze(0, 1).cpu().numpy()

def chip_prob(model, arm, chip_id, mean, std, fraction=0.0):
    """
    Water probability (H, W) for one chip, from its Sentinel-1 image, Sentinel-2 image or both, depending on `arm`.

    Optional: `fraction` of the chip under synthetic clouds, which hide only the Sentinel-2 channels.
    """
    return prob_from_input(model, occluded_input(chip_id, arm, fraction, mean, std)[0])

def make_predictor(model, arm, mean, std, thresh=0.5):
    """The `make_predictor` for `evaluate_with_chip_id`, from a trained model with its arm, normalisation and threshold.

    Each chip's predictor builds its own input from the chip id, so it ignores the vv and vh it receives.
    """
    model = model.to(DEVICE).eval()

    def for_chip(chip_id):
        prob = chip_prob(model, arm, chip_id, mean, std)
        return lambda vv, vh, valid: (prob > thresh) & valid

    return for_chip

def micro_iou(model, arm, chip_ids, thresholds, mean, std, fraction=0.0, desc=""):
    """
    Given a list of thresholds, score the model for the given arm by micro IoU for each threshold over a list of chips.

    Optional: `fraction` of each chip under synthetic clouds, which hide only the Sentinel-2 channels.
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
