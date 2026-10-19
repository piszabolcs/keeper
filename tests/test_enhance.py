# Keeper - denoising, sharpening and the correction chain. Small images keep
# the (slow) NLM fast; the effects are the same as on 24 MP.

import threading

import cv2
import numpy as np
import pytest

import config
from core import enhance
from core.enhance import (
    Cancelled,
    denoise,
    denoise_strength,
    enhance_image,
    load_bgr_full,
    sharpen,
    sharpen_strength,
    sharpen_threshold,
)
from core.noise import estimate_sigma
from core.sharpness import laplacian_variance

SEED = 20260727


def flat(value=128, size=256):
    return np.full((size, size, 3), value, dtype=np.uint8)


def noisy(sigma, value=128, size=256):
    rng = np.random.default_rng(SEED)
    image = np.full((size, size, 3), float(value)) + rng.normal(0.0, sigma, (size, size, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def checkerboard(size=256, step=16):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, step):
        for x in range(0, size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                image[y:y + step, x:x + step] = 220
    return image


def soft_noisy(noise_level):
    rng = np.random.default_rng(SEED)
    base = cv2.GaussianBlur(checkerboard(), (0, 0), 1.4).astype(float)
    return np.clip(base + rng.normal(0.0, noise_level, base.shape), 0, 255).astype(np.uint8)


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# strengths follow the measurement
# ---------------------------------------------------------------------------


def test_denoise_strength_follows_the_noise():
    # a clean frame is not filtered at all: filtering only removes detail there
    assert denoise_strength(config.NOISE_SIGMA_LOW - 0.1) == denoise_strength(0.0) == 0.0
    assert denoise_strength(config.NOISE_SIGMA_HIGH) == denoise_strength(99.0) == config.DENOISE_H_MAX
    assert 0.0 < denoise_strength(3.0) < denoise_strength(6.0) < denoise_strength(9.0)


def test_sharpen_strength_peaks_on_slightly_soft_frames():
    assert sharpen_strength(config.SHARPEN_SCORE_FLOOR - 1.0) == sharpen_strength(0.0) == 0.0
    assert sharpen_strength(None) == 0.0
    peak = sharpen_strength(config.SHARPEN_SCORE_PEAK)
    assert peak == config.SHARPEN_MAX_AMOUNT
    assert peak > sharpen_strength(config.SHARPEN_SCORE_PEAK - 15.0)
    assert peak > sharpen_strength(config.SHARPEN_SCORE_PEAK + 15.0)
    assert sharpen_strength(config.SHARPEN_SCORE_CEILING) == pytest.approx(config.SHARPEN_MIN_AMOUNT)
    assert sharpen_strength(95.0) < config.SHARPEN_MAX_AMOUNT / 3.0


# ---------------------------------------------------------------------------
# the operations, checked with the project's own metrics
# ---------------------------------------------------------------------------


def test_denoising_reduces_noise_and_zero_strength_changes_nothing():
    image = noisy(9.0)
    assert estimate_sigma(gray(denoise(image, 10.0))) < estimate_sigma(gray(image)) * 0.6

    untouched = noisy(5.0)
    assert np.array_equal(denoise(untouched, 0.0), untouched)
    assert np.array_equal(sharpen(untouched, 0.0), untouched)


def test_sharpening_raises_sharpness_stays_in_range_and_spares_flat_areas():
    soft = cv2.GaussianBlur(checkerboard(), (0, 0), 1.5)
    assert laplacian_variance(gray(sharpen(soft, 0.8))) > laplacian_variance(gray(soft))

    result = sharpen(checkerboard(), config.SHARPEN_MAX_AMOUNT)
    assert result.dtype == np.uint8 and result.min() >= 0 and result.max() <= 255

    assert np.abs(sharpen(flat(), config.SHARPEN_MAX_AMOUNT).astype(int) - 128).max() <= 1


def test_the_threshold_follows_the_residual_noise():
    assert sharpen_threshold(flat()) == pytest.approx(config.SHARPEN_THRESHOLD)
    assert sharpen_threshold(noisy(9.0)) > config.SHARPEN_THRESHOLD


def test_tiled_processing_equals_one_pass(monkeypatch):
    """T-11: tiles with overlap give the same pixels as the untiled operation."""
    image = soft_noisy(8.0)
    whole_denoised, whole_sharpened = denoise(image, 4.0), sharpen(image, 0.8, 3.0)

    monkeypatch.setattr(config, "ENHANCE_TILE", 96)
    assert np.array_equal(denoise(image, 4.0), whole_denoised)
    assert np.array_equal(sharpen(image, 0.8, 3.0), whole_sharpened)


def test_a_set_cancel_event_stops_between_tiles(monkeypatch):
    monkeypatch.setattr(config, "ENHANCE_TILE", 64)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(Cancelled):
        enhance_image(soft_noisy(8.0), 8.0, 65.0, cancel=cancel)


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------


def test_a_noisy_soft_frame_gets_both_steps():
    image = soft_noisy(8.0)
    result = enhance_image(image, sigma=8.0, score=65.0, brightness_percent=0)

    assert result.denoise_h > 0.0 and result.sharpen_amount > 0.0
    assert laplacian_variance(gray(result.image)) > laplacian_variance(gray(image))
    assert result.image.shape == image.shape and result.image.dtype == np.uint8


def test_the_denoise_step_always_lowers_the_noise():
    for level in (3.0, 5.0, 8.0, 12.0):
        image = soft_noisy(level)
        assert estimate_sigma(gray(denoise(image, denoise_strength(level)))) < estimate_sigma(gray(image))


def test_sharpening_a_noise_free_frame_raises_the_estimate_a_known_limitation():
    """The estimator reads sharpened fine detail as noise, so the full chain
    can only promise "not noticeably noisier", not "less noisy"."""
    clean_soft = cv2.GaussianBlur(checkerboard(), (0, 0), 1.4)
    sharpened = sharpen(clean_soft, 0.69, sharpen_threshold(clean_soft))
    assert estimate_sigma(gray(sharpened)) > estimate_sigma(gray(clean_soft))


def test_the_chain_never_makes_a_frame_noticeably_noisier():
    for level in (3.0, 5.0, 8.0, 12.0):
        image = soft_noisy(level)
        result = enhance_image(image, sigma=level, score=65.0, brightness_percent=0).image
        assert estimate_sigma(gray(result)) < estimate_sigma(gray(image)) * 1.25, level

    image = soft_noisy(12.0)
    result = enhance_image(image, sigma=12.0, score=65.0, brightness_percent=0).image
    assert estimate_sigma(gray(result)) < estimate_sigma(gray(image))


def test_denoise_then_sharpen_is_the_right_order():
    """Sharpening first would amplify the grain the filter then has to fight
    and smear real detail; the right order ends closer to the clean original."""
    rng = np.random.default_rng(SEED)
    clean = checkerboard().astype(float)
    image = np.clip(clean + rng.normal(0.0, 8.0, clean.shape), 0, 255).astype(np.uint8)

    right = enhance_image(image, sigma=8.0, score=65.0, brightness_percent=0).image
    reversed_order = denoise(sharpen(image, 0.8), 10.0)

    assert np.mean(np.abs(right.astype(float) - clean)) < np.mean(np.abs(reversed_order.astype(float) - clean))


def test_clean_sharp_or_unmeasured_frames_are_left_alone():
    result = enhance_image(checkerboard(), sigma=0.5, score=98.0, brightness_percent=0)
    assert result.denoise_h == 0.0 and result.sharpen_amount < config.SHARPEN_MAX_AMOUNT / 3.0

    original = checkerboard()
    result = enhance_image(original, sigma=None, score=None, brightness_percent=0)
    assert result.denoise_h == result.sharpen_amount == 0.0
    assert np.array_equal(result.image, original)


# ---------------------------------------------------------------------------
# full-resolution loading
# ---------------------------------------------------------------------------


def test_loading_is_full_resolution_bgr_and_upright(tmp_path):
    from PIL import Image

    large = np.zeros((1500, 2400, 3), dtype=np.uint8)
    large[:, 1200:] = 200
    Image.fromarray(large).save(tmp_path / "large.jpg", quality=95)
    assert load_bgr_full(tmp_path / "large.jpg").shape == (1500, 2400, 3)

    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[:, :, 0] = 230
    Image.fromarray(red).save(tmp_path / "red.jpg", quality=100, subsampling=0)
    blue, _, r = load_bgr_full(tmp_path / "red.jpg")[32, 32]
    assert r > 200 and blue < 40

    portrait = Image.fromarray(np.zeros((200, 100, 3), dtype=np.uint8))
    exif = portrait.getexif()
    exif[0x0112] = 6
    portrait.save(tmp_path / "rotated.jpg", exif=exif)
    assert load_bgr_full(tmp_path / "rotated.jpg").shape[:2] == (100, 200)


# ---------------------------------------------------------------------------
# tone and colour inside the chain
# ---------------------------------------------------------------------------


def _dark():
    return (checkerboard().astype(np.float32) * 0.28).astype(np.uint8)


def test_the_chain_brightens_a_dark_tinted_frame():
    image = np.clip(_dark().astype(np.float32) * [1.3, 1.0, 0.75], 0, 255).astype(np.uint8)
    result = enhance_image(image, sigma=3.0, score=65.0)

    assert result.tone_changed and result.gamma < 1.0
    assert np.median(gray(result.image)) > np.median(gray(image))


def test_brightness_and_colour_switch_independently():
    off = enhance_image(_dark(), sigma=3.0, score=65.0, brightness_percent=0, color_percent=0)
    assert not off.tone_changed and off.gains == (1.0, 1.0, 1.0)
    assert off.gamma == pytest.approx(1.0) and off.clahe == 0.0 and off.saturation == pytest.approx(1.0)

    colour_only = enhance_image(_dark(), sigma=3.0, score=65.0, brightness_percent=0, color_percent=100)
    assert colour_only.gamma == pytest.approx(1.0)
    assert colour_only.clahe > 0.0 or colour_only.saturation > 1.0

    brightness_only = enhance_image(_dark(), sigma=3.0, score=65.0, brightness_percent=100, color_percent=0)
    assert brightness_only.gamma != pytest.approx(1.0)
    assert brightness_only.clahe == 0.0 and brightness_only.saturation == pytest.approx(1.0)


def test_noise_is_measured_again_after_brightening():
    rng = np.random.default_rng(SEED)
    base = checkerboard().astype(np.float32) * 0.25
    image = np.clip(base + rng.normal(0.0, 2.0, base.shape), 0, 255).astype(np.uint8)

    result = enhance_image(image, sigma=2.0, score=65.0)

    assert result.sigma > 2.0 and result.denoise_h > denoise_strength(2.0)


def test_a_tone_only_change_still_counts_as_a_change():
    result = enhance_image((checkerboard().astype(np.float32) * 0.3).astype(np.uint8), sigma=0.2, score=98.0)
    assert result.denoise_h == 0.0 and result.tone_changed and result.anything_changed


def test_peak_memory_stays_bounded_on_a_24_mp_frame():
    """C-6: tone LUTs + tiling keep a 24 MP frame far below the old +2 GB."""
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    rng = np.random.default_rng(1)
    image = cv2.resize(rng.integers(0, 255, (600, 400, 3), dtype=np.uint8), (6000, 4000))
    before = process.memory_info().rss
    peak = [before]
    done = threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            done.wait(0.005)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        result = enhance_image(image, sigma=1.0, score=98.0, denoise_percent=0)
    finally:
        done.set()
        sampler.join()

    assert result.image.shape == image.shape
    assert peak[0] - before < 500 * 1024 * 1024, (peak[0] - before) / 2 ** 20
