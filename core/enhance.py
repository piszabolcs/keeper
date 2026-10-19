# Keeper - denoising, sharpening and the whole correction chain.
#
# The MEASURED state sets the strength, not a global slider: on a clean,
# sharp frame both operations are pure loss (denoising removes detail,
# sharpening draws halos), so the right answer there is to do nothing.
# Images are OpenCV BGR uint8 arrays throughout.
#
# Memory: large frames are denoised and sharpened in overlapping tiles and
# the tone steps use lookup tables, so a 24 MP frame needs a few hundred MB
# instead of ~2 GB (C-6). Tiling also gives cancellation points.

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from core import learned_denoise, noise, raw_loader, tone


class Cancelled(Exception):
    """Raised between tiles when the caller's cancel event is set."""


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def load_bgr_full(path):
    """
    Full-resolution colour image for corrections (BGR). Unlike measuring, no
    downscaling and, for RAW, a real demosaic: this is what the user gets.
    """
    if raw_loader.is_raw(path):
        rgb = raw_loader.demosaic_rgb_full(path)
    else:
        with Image.open(path) as img:
            # apply the EXIF rotation, or the saved copy would lie on its side
            img = ImageOps.exif_transpose(img)
            rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _ramp(value, low, high):
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def denoise_strength(sigma):
    """NLM strength for a measured noise sigma; exactly 0 below NOISE_SIGMA_LOW."""
    if sigma is None or sigma < config.NOISE_SIGMA_LOW:
        return 0.0
    position = _ramp(sigma, config.NOISE_SIGMA_LOW, config.NOISE_SIGMA_HIGH)
    return config.DENOISE_H_MIN + position * (config.DENOISE_H_MAX - config.DENOISE_H_MIN)


def sharpen_strength(score):
    """
    Sharpening amount for a score. Zero below FLOOR (the detail is not there,
    sharpening would only amplify noise), the most at PEAK (slightly soft),
    back down to the minimum at CEILING (already sharp - halos).
    """
    if score is None or score < config.SHARPEN_SCORE_FLOOR:
        return 0.0

    if score <= config.SHARPEN_SCORE_PEAK:
        position = _ramp(score, config.SHARPEN_SCORE_FLOOR, config.SHARPEN_SCORE_PEAK)
        low = config.SHARPEN_MIN_AMOUNT
        return low + position * (config.SHARPEN_MAX_AMOUNT - low)

    position = _ramp(score, config.SHARPEN_SCORE_PEAK, config.SHARPEN_SCORE_CEILING)
    return config.SHARPEN_MAX_AMOUNT - position * (
        config.SHARPEN_MAX_AMOUNT - config.SHARPEN_MIN_AMOUNT
    )


def _tiled(image, operation, cancel=None):
    """
    Runs operation on overlapping tiles and stitches the tile interiors. With
    an overlap wider than the operation's reach (NLM: 13 px, unsharp mask:
    5 px) the result equals the untiled one.
    """
    height, width = image.shape[:2]
    size, margin = config.ENHANCE_TILE, config.ENHANCE_TILE_OVERLAP
    _check(cancel)
    if height <= size and width <= size:
        return operation(image)

    result = np.empty_like(image)
    for y in range(0, height, size):
        for x in range(0, width, size):
            _check(cancel)
            y0, x0 = max(0, y - margin), max(0, x - margin)
            y1, x1 = min(height, y + size + margin), min(width, x + size + margin)
            done = operation(np.ascontiguousarray(image[y0:y1, x0:x1]))
            h, w = min(size, height - y), min(size, width - x)
            result[y:y + h, x:x + w] = done[y - y0:y - y0 + h, x - x0:x - x0 + w]
    return result


def denoise(bgr, h, cancel=None):
    """
    Non-local means: each pixel is averaged with pixels whose NEIGHBOURHOODS
    look alike, so edges survive and random noise averages out. Chroma is
    filtered harder (DENOISE_COLOR_FACTOR): colour blotches carry no detail.
    """
    if h <= 0.0:
        return bgr

    def run(part):
        return cv2.fastNlMeansDenoisingColored(
            part, None, float(h), float(h * config.DENOISE_COLOR_FACTOR),
            config.DENOISE_TEMPLATE_SIZE, config.DENOISE_SEARCH_SIZE,
        )

    return _tiled(bgr, run, cancel)


def sharpen_threshold(bgr):
    """
    Unsharp-mask threshold for this frame: at least the fixed floor, more if
    the RESIDUAL noise asks for it - otherwise sharpening re-amplifies the
    grain the denoiser just removed (only edges 3 sigma above the noise count).
    """
    residual = noise.sigma_from_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return max(config.SHARPEN_THRESHOLD, config.SHARPEN_NOISE_GUARD * residual)


def _unsharp(bgr, amount, threshold):
    original = bgr.astype(np.float32)
    detail = original - cv2.GaussianBlur(original, (0, 0), config.SHARPEN_RADIUS)
    # SOFT threshold (subtract it from the magnitude): a hard cut would keep
    # noise peaks just above it at full strength - salt-and-pepper grain
    detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
    # clipping is mandatory, uint8 would wrap around
    return np.clip(original + amount * detail, 0, 255).astype(np.uint8)


def sharpen(bgr, amount, threshold=None, cancel=None):
    """Unsharp mask with a soft threshold so flat areas stay untouched."""
    if amount <= 0.0:
        return bgr
    if threshold is None:
        threshold = config.SHARPEN_THRESHOLD
    return _tiled(bgr, lambda part: _unsharp(part, amount, threshold), cancel)


def scaled_strengths(sigma, score, denoise_percent=100, sharpen_percent=100):
    """
    Final (h, amount): the sliders scale the AUTOMATIC strength, so a clean
    frame stays untouched even at 300 % - there is nothing to amplify.
    """
    h = denoise_strength(sigma) * (denoise_percent / 100.0)
    amount = sharpen_strength(score) * (sharpen_percent / 100.0)
    return min(h, config.DENOISE_H_LIMIT), min(amount, config.SHARPEN_AMOUNT_LIMIT)


@dataclass
class EnhanceResult:
    """The corrected image and what each step did (so the UI can say so)."""

    image: object
    sigma: float | None      # noise level actually used
    denoise_h: float
    sharpen_amount: float
    gains: tuple             # white balance multipliers (B, G, R)
    gamma: float             # brightness curve
    clahe: float = 0.0       # local contrast clip
    saturation: float = 1.0  # saturation multiplier
    learned: bool = False    # the learned (SCUNet) denoiser ran

    @property
    def tone_changed(self):
        return (
            any(abs(g - 1.0) > 1e-6 for g in self.gains)
            or abs(self.gamma - 1.0) > 1e-6
            or self.clahe > 0.0
            or abs(self.saturation - 1.0) > 1e-6
        )

    @property
    def anything_changed(self):
        return self.denoise_h > 0.0 or self.sharpen_amount > 0.0 or self.tone_changed


def enhance_image(
    bgr, sigma, score, denoise_percent=100, sharpen_percent=100, brightness_percent=100,
    iso=None, wb_percent=0, learned=False, color_percent=100,
    stats=None, noise_crop=None, cancel=None,
):
    """
    The whole correction of one image. The ORDER matters:
      1. brightness / white balance - global; brightening also raises the
         noise, so the noise is re-measured afterwards
      2. denoising (classic NLM or the learned network)
      3. local contrast and saturation - both amplify noise, so after (2)
      4. sharpening - last, or it would amplify the noise for (2) to fight

    stats is the bounded-resolution copy of the WHOLE frame that all tone
    statistics are measured on, noise_crop the frame's native-resolution
    centre crop for the noise re-measurement. Both default to the image
    itself; the preview passes those of the full frame, so a preview crop is
    corrected with exactly the parameters the saved copy gets.
    """
    if stats is None:
        stats = tone.stats_image(bgr)
    if noise_crop is None:
        noise_crop = noise.center_crop(bgr)

    gains, gamma, toned_stats = tone.plan_exposure(stats, brightness_percent, wb_percent)
    result = tone.apply_gamma(tone.apply_gains(bgr, gains), gamma)

    brightening = 1.0
    if any(abs(g - 1.0) > 1e-6 for g in gains) or abs(gamma - 1.0) > 1e-6:
        # the earlier noise reading is stale: brightening raises the grain too
        toned_crop = tone.apply_gamma(tone.apply_gains(noise_crop, gains), gamma)
        sigma = noise.estimate_sigma(cv2.cvtColor(toned_crop, cv2.COLOR_BGR2GRAY))
        before = max(float(np.median(cv2.cvtColor(stats, cv2.COLOR_BGR2GRAY))), 1.0)
        brightening = float(np.median(cv2.cvtColor(toned_stats, cv2.COLOR_BGR2GRAY))) / before

    # the ISO is an independent hint of how much noise there can be
    sigma = noise.iso_capped_sigma(sigma, iso, brightening)
    h, amount = scaled_strengths(sigma, score, denoise_percent, sharpen_percent)

    # the learned network only runs where the classic filter would run too
    learned_ran = bool(learned and h > 0.0 and learned_denoise.available())
    if learned_ran:
        result = learned_denoise.denoise(result, cancel)
    else:
        result = denoise(result, h, cancel)

    clip, saturation = 0.0, 1.0
    if color_percent:
        # the noise LEFT after denoising limits how far CLAHE may go
        residual = noise.sigma_from_gray(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        clip, saturation = tone.plan_color(toned_stats, color_percent, residual)
        result = tone.apply_saturation(tone.apply_clahe(result, clip), saturation)

    result = sharpen(result, amount, sharpen_threshold(result), cancel)

    return EnhanceResult(
        image=result, sigma=sigma, denoise_h=h, sharpen_amount=amount,
        gains=gains, gamma=gamma, clahe=clip, saturation=saturation, learned=learned_ran,
    )
