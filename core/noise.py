# Keeper - how much noise is in a picture.
#
# Denoising follows the MEASURED noise, not the ISO: an underexposed ISO 400
# frame pushed in post can be noisier than a well exposed ISO 3200 one.

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config
from core import exif, raw_loader

# Immerkaer's mask: zero response on flat areas AND on linear gradients (a
# soft sky), so ideally only noise remains. The 6 and sqrt(pi/2) factors come
# from the derivation of the method.
IMMERKAER_MASK = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])

# The estimate is taken per block and the low percentile is used (see
# estimate_sigma).
BLOCK_SIZE = 32
BLOCK_PERCENTILE = 25


@dataclass
class NoiseResult:
    """Noise of one file. On error sigma is None and error is set."""

    path: Path
    sigma: float | None   # estimated noise std on a 0-255 scale
    iso: int | None       # shooting ISO if readable (informational)
    error: str | None


def _raw_sigma(gray):
    """Immerkaer estimate of one region (the border row/column is dropped)."""
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 0.0

    response = cv2.filter2D(
        gray.astype(np.float64), -1, IMMERKAER_MASK, borderType=cv2.BORDER_REPLICATE
    )
    inner = np.abs(response[1:-1, 1:-1])
    return math.sqrt(math.pi / 2.0) * float(inner.sum()) / (6.0 * (width - 2) * (height - 2))


def estimate_sigma(gray):
    """
    Noise std of a grey image.

    The mask also responds to fine DETAIL (grass, hair, fabric). One value for
    the whole frame would make a sharp, textured photo look noisy and get
    over-filtered. So the frame is split into blocks and a LOW percentile of
    the per-block estimates is used: almost every photo has some smooth area
    (sky, wall, out-of-focus background) where noise can be read cleanly.
    Limitation: a frame without any smooth patch is overestimated.
    """
    height, width = gray.shape[:2]
    if height < BLOCK_SIZE or width < BLOCK_SIZE:
        return _raw_sigma(gray)

    estimates = [
        _raw_sigma(gray[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE])
        for y in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE)
        for x in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE)
    ]
    return float(np.percentile(estimates, BLOCK_PERCENTILE))


def center_crop(image):
    """
    A NOISE_CROP square from the middle of the image at NATIVE resolution
    (grey or colour). Downscaling averages noise away - a 1024 px copy of an
    ISO 12800 frame looks clean - so here we crop instead of scaling. The
    centre avoids vignetted corners.
    """
    height, width = image.shape[:2]
    size = config.NOISE_CROP
    if height <= size and width <= size:
        return image

    y = max(0, (height - size) // 2)
    x = max(0, (width - size) // 2)
    return image[y:y + min(size, height), x:x + min(size, width)]


def sigma_from_gray(gray):
    """Noise of an already loaded grey image (no second decode of the file)."""
    return estimate_sigma(center_crop(gray))


# ISO lives in the shared EXIF reader. None is normal (lost on export).
read_iso = exif.read_iso


def iso_capped_sigma(sigma, iso, brightening=1.0):
    """
    Caps the measured noise by what the ISO makes plausible.

    The estimator can mistake very fine detail for noise; the ISO is an
    independent fact about how much noise there CAN be. brightening is how
    much our own tone correction raised the brightness - that extra noise is
    real, so the cap relaxes accordingly.
    """
    if sigma is None or iso is None or iso <= 0:
        return sigma

    cap = config.ISO_SIGMA_CAP_BASE * (iso / config.ISO_REFERENCE) ** config.ISO_SIGMA_CAP_EXPONENT
    return min(sigma, cap * max(1.0, brightening))


def measure_noise(path):
    """
    Noise of a file. Never raises. RAW files are measured on the embedded
    preview (after the camera's own noise reduction) - the same image the
    corrections work on.
    """
    path = Path(path)
    try:
        if raw_loader.is_raw(path):
            gray, _ = raw_loader.load_gray_from_raw(path)
        else:
            with Image.open(path) as img:
                # deliberately no draft(): DCT scaling would remove the noise
                gray = np.array(img.convert("L"))
    except Exception as exc:
        return NoiseResult(path=path, sigma=None, iso=None, error=str(exc))

    return NoiseResult(path=path, sigma=sigma_from_gray(gray), iso=read_iso(path), error=None)
