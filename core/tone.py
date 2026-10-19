# Keeper - brightness and colour correction.
#
# Same principle as denoising: the correction follows the MEASURED state, and
# a well exposed, neutral frame is not touched at all. Colour temperature can
# be INTENT (a sunset, candle light), so every step is strictly limited.
#
# Statistics are measured on a bounded-resolution copy of the WHOLE frame
# (stats_image); the resulting parameters are applied to whatever is being
# processed - the full image when saving, a crop in the preview. That is what
# makes the preview show the same correction the saved copy gets.
#
# The steps together guarantee that nothing new clips: white balance never
# brightens (all gains <= 1), the gamma curve keeps 0 and 255 in place, and
# saturation leaves the HSV value channel (the max of R, G, B) untouched.

from dataclasses import dataclass

import cv2
import numpy as np

import config


def stats_image(bgr):
    """A copy of the whole frame scaled to TONE_STATS_RESOLUTION (or the frame)."""
    height, width = bgr.shape[:2]
    longer = max(height, width)
    if longer <= config.TONE_STATS_RESOLUTION:
        return bgr
    ratio = config.TONE_STATS_RESOLUTION / longer
    size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def _gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _lut(scale):
    return np.clip(np.arange(256, dtype=np.float32) * scale, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# white balance
# ---------------------------------------------------------------------------


def white_balance_gains(bgr):
    """
    (blue, green, red) multipliers that neutralise a colour cast, using the
    shades-of-grey estimate (Minkowski p-norm per channel). p=6 sits between
    grey-world (p=1: assumes the average colour is grey - wrong for a green
    forest) and white-patch (p=inf: one specular highlight decides).
    """
    p = config.WB_MINKOWSKI_P
    image = bgr.astype(np.float64)  # the 6th power overflows anything smaller
    norms = [float(np.mean(image[:, :, ch] ** p)) ** (1.0 / p) for ch in range(3)]

    mean = sum(norms) / 3.0
    if mean <= 1e-6:
        return (1.0, 1.0, 1.0)
    return tuple(mean / max(norm, 1e-6) for norm in norms)


def wb_deviation(gains):
    """Largest distance from 1.0: how coloured the cast is."""
    return max(abs(g - 1.0) for g in gains)


def limited_gains(gains, percent=100):
    """
    The multipliers actually applied:
      1. below WB_MIN_DEVIATION nothing (estimator noise, invisible cast)
      2. the user's percent scales the distance from 1.0
      3. hard limit WB_GAIN_LIMIT (protects intentional warm/cool tones)
      4. normalised so the largest gain is exactly 1.0: a gain above 1.0 clips
         highlights (measured: 6.7 % -> 9.9 % clipped area from a 1.038 green
         gain). The RATIO, which is what white balance is about, is unchanged;
         the exposure step gives the brightness back without clipping.
    """
    if wb_deviation(gains) < config.WB_MIN_DEVIATION:
        return (1.0, 1.0, 1.0)

    ratio = percent / 100.0
    upper = config.WB_GAIN_LIMIT
    limited = [min(upper, max(1.0 / upper, 1.0 + (g - 1.0) * ratio)) for g in gains]

    largest = max(limited)
    if largest <= 1.0:
        return tuple(limited)
    return tuple(g / largest for g in limited)


def apply_gains(bgr, gains):
    """Applies the channel multipliers (clipped) through per-channel LUTs."""
    if all(abs(g - 1.0) < 1e-6 for g in gains):
        return bgr
    table = np.stack([_lut(g) for g in gains], axis=-1).reshape(256, 1, 3)
    return cv2.LUT(bgr, table)


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------


def exposure_gamma(gray, percent=100, base_median=None):
    """
    Gamma that brings the brightness MEDIAN into the acceptable band
    (1.0 = nothing to do). Dark frames are lifted only to the bottom of the
    band, bright ones pulled back only to its top. base_median is the
    brightness before white balance (which can only darken), so the
    brightness is judged on the original. Gamma keeps 0 and 255 in place, so
    highlights never clip.
    """
    median = float(np.median(gray))
    if median <= 1.0 or median >= 254.0:
        return 1.0

    base = median if base_median is None else float(base_median)
    target = min(config.EXPOSURE_TARGET_HIGH, max(config.EXPOSURE_TARGET_LOW, base))
    if abs(target - median) < 0.5:
        return 1.0

    gamma = np.log(target / 255.0) / np.log(median / 255.0)
    gamma = 1.0 + (gamma - 1.0) * (percent / 100.0)

    limit = config.EXPOSURE_GAMMA_LIMIT
    return float(min(limit, max(1.0 / limit, gamma)))


def apply_gamma(bgr, gamma):
    """Gamma curve through a 256-entry lookup table."""
    if abs(gamma - 1.0) < 1e-6:
        return bgr
    table = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.power(table, gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


# ---------------------------------------------------------------------------
# local contrast (CLAHE)
# ---------------------------------------------------------------------------


def contrast_span(bgr):
    """95th minus 5th percentile of L* (0-255): how much of the tonal range is used."""
    lightness = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    low, high = np.percentile(lightness, (5, 95))
    return float(high - low)


def clahe_clip(bgr, percent=100, sigma=None):
    """
    CLAHE clip limit for this frame, 0.0 = leave it alone. Only flat frames
    (haze, flat light) get it, scaled by the slider and capped by
    CLAHE_CLIP_MAX AFTER scaling. CLAHE also amplifies noise
    (sigma_out ~ (1 + 0.7 clip) sigma_in), so the clip is limited to keep the
    output noise under CLAHE_NOISE_CEILING.
    """
    span = contrast_span(bgr)
    if span >= config.CONTRAST_TARGET_SPAN or span < config.CONTRAST_MIN_SPAN:
        return 0.0

    flatness = (config.CONTRAST_TARGET_SPAN - span) / (
        config.CONTRAST_TARGET_SPAN - config.CONTRAST_FLAT_SPAN
    )
    flatness = min(1.0, max(0.0, flatness))
    clip = min(config.CLAHE_CLIP_MAX, config.CLAHE_CLIP_MAX * flatness * (percent / 100.0))

    if sigma is None:
        from core import noise

        sigma = noise.sigma_from_gray(_gray(bgr))
    if sigma and sigma > 0.0:
        room = (config.CLAHE_NOISE_CEILING / sigma - 1.0) / config.CLAHE_NOISE_GAIN
        clip = min(clip, max(0.0, room))

    # below the minimum CLAHE does nothing visible - do not report a fake step
    return clip if clip >= config.CLAHE_CLIP_MIN else 0.0


def apply_clahe(bgr, clip):
    """Contrast-limited equalisation of the L* channel only (colours stay)."""
    if clip <= 0.0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    equaliser = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=config.CLAHE_GRID)
    lab[:, :, 0] = equaliser.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# saturation
# ---------------------------------------------------------------------------


def saturation_level(bgr):
    """Median HSV saturation of the usefully bright pixels (None if there are none)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    usable = (hsv[:, :, 2] >= config.SATURATION_MIN_VALUE) & (
        hsv[:, :, 2] <= config.SATURATION_MAX_VALUE
    )
    if not usable.any():
        return None
    return float(np.median(hsv[:, :, 1][usable]))


def saturation_scale(bgr, percent=100):
    """
    Saturation multiplier, 1.0 = untouched. Only lifts dull frames, never
    desaturates. Limited twice: the automatic need is capped first (the slider
    scales that judgement), then the result is capped again so no slider
    position can exceed SATURATION_SCALE_LIMIT.
    """
    level = saturation_level(bgr)
    if level is None or level >= config.SATURATION_TARGET:
        return 1.0

    automatic = min(config.SATURATION_TARGET / max(level, 1.0), config.SATURATION_SCALE_LIMIT)
    scaled = 1.0 + (automatic - 1.0) * (percent / 100.0)
    return min(scaled, config.SATURATION_SCALE_LIMIT)


def apply_saturation(bgr, scale):
    """Scales the HSV S channel through a lookup table; V is untouched."""
    if abs(scale - 1.0) < 1e-6:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], _lut(scale))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# planning on the statistics image, applying anywhere
# ---------------------------------------------------------------------------


def measure(bgr):
    """(raw gains, brightness median) of a frame."""
    return white_balance_gains(bgr), float(np.median(_gray(bgr)))


def plan_exposure(stats, percent=100, wb_percent=0):
    """
    (gains, gamma, toned stats image) measured on the statistics image.
    White balance first (a cast shifts the median), then the gamma measured on
    the balanced image against the ORIGINAL median.
    """
    base_median = float(np.median(_gray(stats)))
    gains = limited_gains(white_balance_gains(stats), wb_percent)
    balanced = apply_gains(stats, gains)
    gamma = exposure_gamma(_gray(balanced), percent, base_median)
    return gains, gamma, apply_gamma(balanced, gamma)


def plan_color(toned_stats, percent=100, sigma=None):
    """(CLAHE clip, saturation scale) measured on the toned statistics image."""
    clip = clahe_clip(toned_stats, percent, sigma)
    scale = saturation_scale(apply_clahe(toned_stats, clip), percent)
    return clip, scale


@dataclass
class ToneResult:
    """The corrected image and what each step did."""

    image: object
    gains: tuple       # white balance (B, G, R)
    gamma: float       # brightness curve
    clahe: float       # local contrast clip, 0.0 = skipped
    saturation: float  # saturation multiplier, 1.0 = untouched

    @property
    def changed(self):
        return (
            any(abs(g - 1.0) > 1e-6 for g in self.gains)
            or abs(self.gamma - 1.0) > 1e-6
            or self.clahe > 0.0
            or abs(self.saturation - 1.0) > 1e-6
        )


def correct(bgr, percent=100, wb_percent=0, color_percent=100, sigma=None, stats=None):
    """
    The whole tone and colour correction: white balance -> brightness ->
    local contrast -> saturation. White balance is OFF by default (wb_percent
    0): on 136 reference frames it neutralised warm subjects and blue skies
    more often than it fixed real casts.
    """
    if stats is None:
        stats = stats_image(bgr)

    gains, gamma, toned_stats = plan_exposure(stats, percent, wb_percent)
    result = apply_gamma(apply_gains(bgr, gains), gamma)

    clip, scale = 0.0, 1.0
    if color_percent:
        clip, scale = plan_color(toned_stats, color_percent, sigma)
        result = apply_saturation(apply_clahe(result, clip), scale)

    return ToneResult(image=result, gains=gains, gamma=gamma, clahe=clip, saturation=scale)


def colorize(bgr, percent=100, sigma=None, stats=None):
    """
    Local contrast and saturation only; returns (image, clip, scale). Both
    amplify noise, which is why the enhancement runs them AFTER denoising.
    """
    clip, scale = plan_color(bgr if stats is None else stats, percent, sigma)
    return apply_saturation(apply_clahe(bgr, clip), scale), clip, scale
