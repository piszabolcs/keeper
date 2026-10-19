# Keeper - the sharpness score.
#
# Three classic focus metrics are combined because none is reliable alone.
# Each metric is normalised by the frame's own contrast and has the noise
# contribution removed; without that the raw energy measured the LIGHT, not
# the focus (68 sunny frames all landed between 78 and 99, soft ones on top).

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from core import exif, noise, raw_loader, series

_weights = config.WEIGHT_LAPLACIAN + config.WEIGHT_TENENGRAD + config.WEIGHT_BRENNER
if abs(_weights - 1.0) > 1e-6:
    raise ValueError("the three sharpness weights in config.py must sum to 1.0, not %s" % _weights)


@dataclass
class SharpnessResult:
    """Measurement of one shot. On error score is None and error is set."""

    path: Path
    score: float | None
    laplacian: float | None
    tenengrad: float | None
    brenner: float | None
    low_resolution: bool
    error: str | None
    # where the measured pixels came from (jpeg / raw-preview / raw-demosaic)
    source: str = raw_loader.SOURCE_JPEG
    # every file that moves with the shot (two for a RAW+JPEG pair); never empty
    files: list = None
    # the photographer's decision, stronger than the threshold:
    # None = threshold decides, True = always keep, False = always reject
    override: bool = None
    # surplus member of a series. A separate flag, not a third override state:
    # a duplicate is a FINE frame that goes to duplicates/, not rejected/.
    duplicate: bool = False
    # for series detection; both come for free during measuring
    phash: int = None        # perceptual hash, see core/series.py
    captured: float = None   # EXIF capture time (unix), None if unknown
    # face analysis (core.faces.FaceReport); filled by the optional face pass
    faces: object = None
    # the measured score before the shooting-condition adjustment, and the
    # adjustment itself (positive = stricter); shown on the card
    raw_score: float = None
    expectation: float = 0.0
    # shooting data (core.exif.ShotInfo) and header-only hints
    shot: object = None
    warnings: list = None
    # thumbnail prepared by the GUI worker (a QImage); not used by the core
    thumbnail: object = None

    def __post_init__(self):
        if self.files is None:
            self.files = [self.path]
        if self.warnings is None:
            self.warnings = []


def is_kept(result, threshold):
    """
    Does the shot stay at this threshold? Decided in ONE place for the card
    colour, the counters and the actual move:
      1. an unreadable shot is never touched (kept)
      2. a manual decision beats the threshold
      3. otherwise the threshold decides
    """
    if result.score is None:
        return True
    if result.override is not None:
        return result.override
    return result.score >= threshold


def clear_overrides(results):
    """Removes every manual decision; returns how many shots it touched."""
    count = 0
    for result in results:
        if result.override is not None:
            result.override = None
            count += 1
    return count


def normalize_log(value, lo, hi):
    """
    Maps a metric to 0-100 on a LOG scale: the metrics span orders of
    magnitude, a linear scale would cram every sharp frame into 95-100.
    Zero and negative values (a perfectly flat area after noise removal) get 0.
    """
    if hi <= lo:
        raise ValueError("HI must be larger than LO")
    if value <= 0:
        return 0.0

    position = (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return 100.0 * min(1.0, max(0.0, position))


def laplacian_variance(gray):
    """Variance of the Laplacian: large on sharp frames, small on soft ones."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray):
    """Mean squared Sobel gradient: how strong the edges are."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def brenner(gray):
    """Squared difference of pixels two apart horizontally; catches motion blur."""
    if gray.shape[1] < 3:
        return 0.0
    image = gray.astype(np.float64)
    diff = image[:, 2:] - image[:, :-2]
    return float(np.mean(diff * diff))


def metrics(gray, sigma=None):
    """
    The three contrast-normalised, noise-corrected metrics of a grey image.

    1. Contrast normalisation: every metric is a squared pixel quantity, so
       dividing by the image variance makes it exactly independent of the
       contrast scale (bright sun vs. flat indoor light).
    2. Noise removal: white noise adds sigma^2 x (sum of squared filter
       coefficients) to each metric; that is subtracted.

    sigma can be passed in: on a small crop (a face) the estimate is unreliable,
    while noise is a property of the whole frame.
    """
    image = gray.astype(np.float64)
    if sigma is None:
        sigma = noise.estimate_sigma(gray)
    noise_power = float(sigma) ** 2

    contrast = max(float(image.std()), config.CONTRAST_FLOOR)
    denominator = contrast * contrast

    lap = laplacian_variance(image) - config.NOISE_LAPLACIAN_GAIN * noise_power
    ten = tenengrad(image) - config.NOISE_TENENGRAD_GAIN * noise_power
    bre = brenner(image) - config.NOISE_BRENNER_GAIN * noise_power

    return max(0.0, lap) / denominator, max(0.0, ten) / denominator, max(0.0, bre) / denominator


def score_parts(gray, sigma=None):
    """
    (total, laplacian, tenengrad, brenner) on the 0-100 scale. Used for the
    whole frame AND for faces, so "frame 95, face 40" stays comparable.
    """
    raw_lap, raw_ten, raw_bre = metrics(gray, sigma)

    lap = normalize_log(raw_lap, config.LAPLACIAN_LO, config.LAPLACIAN_HI)
    ten = normalize_log(raw_ten, config.TENENGRAD_LO, config.TENENGRAD_HI)
    bre = normalize_log(raw_bre, config.BRENNER_LO, config.BRENNER_HI)

    total = (
        config.WEIGHT_LAPLACIAN * lap
        + config.WEIGHT_TENENGRAD * ten
        + config.WEIGHT_BRENNER * bre
    )
    return total, lap, ten, bre


# ---------------------------------------------------------------------------
# expectation from the shooting conditions
# ---------------------------------------------------------------------------


def _ramp(value, low, high):
    """Where value sits between low and high, clipped to 0..1."""
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def expectation_offset(shot):
    """
    How far the measured score is shifted for the shooting conditions.
    POSITIVE = stricter (subtracted), NEGATIVE = more lenient (added).

    Strictness needs EVERY condition to be easy (the smaller factor decides);
    leniency needs ANY condition to be hard (the larger factor decides - adding
    them would reward the same bad light twice). Missing data is neutral.
    """
    if shot is None:
        return 0.0

    margin = exif.shake_margin_stops(shot)
    if margin is None:
        shake_free = shake_risk = 0.0
    else:
        shake_free = _ramp(margin, 0.0, config.STRICT_SHAKE_STOPS)
        shake_risk = _ramp(-margin, 0.0, config.LENIENT_SHAKE_STOPS)

    if not shot.iso:
        clean_light = noise_risk = 0.0
    else:
        # in stops, like the ISO itself behaves
        iso_stop = math.log2(max(shot.iso, 1) / config.ISO_REFERENCE)
        low_stop = math.log2(config.STRICT_MAX_ISO / config.ISO_REFERENCE)
        high_stop = math.log2(config.LENIENT_FULL_ISO / config.ISO_REFERENCE)
        clean_light = 1.0 - _ramp(iso_stop, 0.0, low_stop)
        noise_risk = _ramp(iso_stop, low_stop, high_stop)

    easy = min(shake_free, clean_light)
    hard = max(shake_risk, noise_risk)
    return config.STRICT_PENALTY * easy - config.LENIENT_BONUS * hard


# ---------------------------------------------------------------------------
# loading and measuring
# ---------------------------------------------------------------------------


def load_gray_for_measurement(path):
    """
    (grey array, low_resolution, source) scaled to the work resolution.

    RAW goes through raw_loader, everything else through Pillow; the scaling
    is shared so RAW and JPEG scores stay comparable. draft() lets libjpeg
    downscale while decoding - 4-8x faster than decode + resize.
    """
    if raw_loader.is_raw(path):
        gray, source = raw_loader.load_gray_from_raw(path)
    else:
        with Image.open(path) as img:
            img.draft("L", (config.WORK_RESOLUTION, config.WORK_RESOLUTION))
            # EXIF rotation BEFORE measuring: Brenner is horizontal only
            img = ImageOps.exif_transpose(img)
            gray = np.array(img.convert("L"))
        source = raw_loader.SOURCE_JPEG

    height, width = gray.shape[:2]
    longer = max(height, width)
    # small frames are not upscaled, only flagged (the metrics are means)
    low_resolution = longer < config.WORK_RESOLUTION

    if longer > config.WORK_RESOLUTION:
        ratio = config.WORK_RESOLUTION / longer
        size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return gray, low_resolution, source


def measure_sharpness(path, files=None):
    """
    0-100 sharpness of one shot. files are all files of the shot (two for a
    RAW+JPEG pair). If the primary file cannot be read, the other members are
    tried (a broken JPEG next to a good RAW, C-14); the result then refers to
    the file that was actually measured.

    Never raises: an unreadable shot gets score None and an error, and is
    never moved.
    """
    path = Path(path)
    candidates = [path] + [Path(f) for f in (files or []) if Path(f) != path]

    first_error = None
    for candidate in candidates:
        try:
            gray, low_resolution, source = load_gray_for_measurement(candidate)
        except Exception as exc:
            first_error = first_error or str(exc)
            continue
        return _scored(candidate, gray, low_resolution, source, files)

    return SharpnessResult(
        path=path, score=None, laplacian=None, tenengrad=None, brenner=None,
        low_resolution=False, error=first_error, files=files,
    )


def _scored(path, gray, low_resolution, source, files):
    raw_score, lap, ten, bre = score_parts(gray)

    # the header is read ONCE and serves the expectation, the hints and the
    # series time stamp
    shot = exif.read(path)
    offset = expectation_offset(shot)

    return SharpnessResult(
        path=path,
        score=float(min(100.0, max(0.0, raw_score - offset))),
        laplacian=lap,
        tenengrad=ten,
        brenner=bre,
        low_resolution=low_resolution,
        error=None,
        source=source,
        files=files,
        # the hash comes from the pixels already loaded (a 32x32 DCT)
        phash=series.phash(gray),
        # EXIF time only: file times are unreliable after copying (C-13)
        captured=shot.captured,
        raw_score=raw_score,
        expectation=offset,
        shot=shot,
        warnings=exif.warnings(shot),
    )
