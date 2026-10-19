# Keeper - learned (neural network) denoising with SCUNet. OPTIONAL.
#
# Classic NLM smears fine texture together with the grain; a trained network
# separates the two far better. Measured on a real frame with realistic
# synthetic sensor noise (PSNR against the clean original):
#
#     noisy input       17.9 dB
#     classic NLM       20.5 dB
#     NAFNet width32    30.9 dB
#     SCUNet            36.6 dB   <- built in
#
# It needs three things, none of them obvious: PyTorch with CUDA (~2.6 GB),
# an NVIDIA GPU (on a CPU this model takes ~25 s per megapixel) and the 72 MB
# weights file (tools/download_models.py, checked against its SHA-256).
# Without any of them the program works the same with the classic filter, and
# the UI says what is missing.

import hashlib
from pathlib import Path

import cv2
import numpy as np

import config

# loaded model: (network, device) or None
_cache = None
# result of unavailable_reason(); probing for torch is slow, so it is cached
_UNSET = object()
_reason = _UNSET


def weights_path():
    return Path(__file__).resolve().parent.parent / config.LEARNED_MODEL_DIR / config.LEARNED_MODEL_FILE


def _weights_match(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == config.LEARNED_MODEL_SHA256


def unavailable_reason(refresh=False):
    """
    What is missing for the learned denoiser, or None if it can run. A plain
    yes/no is not enough: each missing piece needs a different fix. The
    answer is cached; refresh=True probes again.
    """
    global _reason
    if _reason is not _UNSET and not refresh:
        return _reason

    _reason = _probe()
    return _reason


def _probe():
    try:
        import torch
    except ImportError:
        return (
            "PyTorch is not installed. The README section "
            "\"Learned denoiser\" explains how to add it."
        )

    if not torch.cuda.is_available():
        return (
            "No NVIDIA graphics card (CUDA) available. On a CPU this model "
            "would take about a quarter of an hour per photo, so it is not offered."
        )

    path = weights_path()
    if not path.exists():
        return (
            "The %s weights file (about 72 MB) is missing. Run "
            "tools/download_models.py to fetch it." % config.LEARNED_MODEL_FILE
        )
    if not _weights_match(path):
        return (
            "The %s weights file does not match the published checksum. "
            "Download it again with tools/download_models.py." % config.LEARNED_MODEL_FILE
        )
    return None


def available():
    return unavailable_reason() is None


def _model():
    """Loads (and caches) the network. Only call when available()."""
    global _cache
    if _cache is not None:
        return _cache

    import torch

    from vendor.scunet import SCUNet

    device = torch.device("cuda")
    network = SCUNet(in_nc=3, config=list(config.LEARNED_BLOCKS), dim=config.LEARNED_DIM)
    # weights_only=True: a tensor file, never a pickle that could run code
    network.load_state_dict(
        torch.load(weights_path(), map_location="cpu", weights_only=True), strict=True
    )
    _cache = (network.eval().to(device), device)
    return _cache


def denoise(bgr, cancel=None):
    """
    The learned denoiser on a BGR image, in overlapping TILES: a 42 MP frame
    in one pass would need ~10 GB of VRAM. Each tile is fed with a margin of
    real surroundings and only its interior is kept, so no seams show.
    """
    import torch

    from core.enhance import _check

    network, device = _model()
    height, width = bgr.shape[:2]
    tile, margin = config.LEARNED_TILE, config.LEARNED_TILE_OVERLAP

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    result = np.empty_like(rgb)

    with torch.no_grad():
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                _check(cancel)
                y_end, x_end = min(y + tile, height), min(x + tile, width)
                y0, x0 = max(0, y - margin), max(0, x - margin)
                y1, x1 = min(height, y_end + margin), min(width, x_end + margin)

                part = torch.from_numpy(rgb[y0:y1, x0:x1]).permute(2, 0, 1).unsqueeze(0).to(device)
                out = network(part).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                result[y:y_end, x:x_end] = out[y - y0:y - y0 + (y_end - y), x - x0:x - x0 + (x_end - x)]
                del part, out

    # round, do not truncate (truncation biases the image by -0.5 levels)
    return cv2.cvtColor(np.rint(result * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
