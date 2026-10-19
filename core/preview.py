# Keeper - the correction preview.
#
# Denoising a whole frame takes seconds, too slow for a slider. The preview
# works on a CROP at native resolution instead. Denoising and sharpening are
# local operations, so the crop gets exactly what the full frame gets there.
# The global parts (noise level, white balance, brightness, local contrast,
# saturation) are measured on the WHOLE frame and passed in, so the preview
# does not invent its own correction from the crop (C-4).
#
# Remaining difference, by design: the residual noise that limits CLAHE and
# sets the sharpening threshold is measured on the denoised crop, not on the
# denoised full frame (denoising the full frame for every slider move is what
# the preview avoids). It only matters on very noisy frames.

from dataclasses import dataclass

import cv2
import numpy as np

import config
from core import noise, tone
from core.enhance import enhance_image


@dataclass
class PreviewInput:
    """Everything the preview needs from the full frame."""

    crop: object        # the detailed crop at native resolution (BGR)
    location: tuple     # (x, y) of the crop in the frame
    sigma: float        # noise measured on the full frame
    stats: object       # tone statistics image of the full frame
    noise_crop: object  # native-resolution centre crop of the full frame


def detailed_crop(bgr, size=None):
    """
    (crop, (x, y)) of the most detailed area. Unlike noise measurement (which
    wants FLAT areas), the preview must show where the correction is visible:
    on a patch of clear sky even strong sharpening is invisible. The search
    runs on a decimated copy, so even 60 MP takes milliseconds.
    """
    size = size or config.PREVIEW_CROP
    height, width = bgr.shape[:2]
    crop_h, crop_w = min(size, height), min(size, width)

    if height <= crop_h and width <= crop_w:
        return bgr, (0, 0)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # plain decimation is fine: we only look for a place, not measure
    step = max(1, int(round(max(height, width) / 512.0)))
    small = gray[::step, ::step].astype(np.float32)

    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    small_h, small_w = max(1, crop_h // step), max(1, crop_w // step)
    energy = cv2.boxFilter(gx * gx + gy * gy, -1, (small_w, small_h), normalize=True)

    # zero the border band: the box filter averages over padding there (false
    # peaks), and a crop centred there would stick out of the frame
    half_h, half_w = small_h // 2, small_w // 2
    bottom, right = energy.shape[0] - half_h, energy.shape[1] - half_w
    if bottom > half_h and right > half_w:
        mask = np.zeros_like(energy)
        mask[half_h:bottom, half_w:right] = 1.0
        energy = energy * mask

    _, _, _, peak = cv2.minMaxLoc(energy)
    x = int(min(max(peak[0] * step - crop_w // 2, 0), width - crop_w))
    y = int(min(max(peak[1] * step - crop_h // 2, 0), height - crop_h))
    return bgr[y:y + crop_h, x:x + crop_w], (x, y)


def prepare(bgr, size=None):
    """
    Measures the full frame once: the detailed crop plus the frame-wide noise
    and tone statistics. The noise comes from the whole frame on purpose: the
    detailed crop would make the estimator mistake detail for noise.
    """
    crop, location = detailed_crop(bgr, size)
    return PreviewInput(
        crop=crop,
        location=location,
        sigma=noise.sigma_from_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
        stats=tone.stats_image(bgr),
        noise_crop=noise.center_crop(bgr).copy(),
    )


def render(source, score, denoise_percent=100, sharpen_percent=100, brightness_percent=100,
           iso=None, wb_percent=0, learned=False, color_percent=100):
    """
    The correction of the preview crop - the SAME enhance_image the save uses,
    fed with the full frame's statistics.
    """
    return enhance_image(
        source.crop, source.sigma, score, denoise_percent, sharpen_percent,
        brightness_percent, iso, wb_percent, learned, color_percent,
        stats=source.stats, noise_crop=source.noise_crop,
    )
