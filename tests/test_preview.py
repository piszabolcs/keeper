# Keeper - the preview crop and the preview correction.

import cv2
import numpy as np
import pytest

import config
from core import noise, preview, tone
from core.enhance import enhance_image


def _one_detailed_patch(x, y, size=(900, 1200)):
    """Flat grey with a single textured square - the only detail in the frame."""
    image = np.full((size[0], size[1], 3), 120, np.uint8)
    pattern = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    image[y:y + 300, x:x + 300] = pattern
    return image


def _overlap(a, a_size, b, b_size):
    w = max(0, min(a[0] + a_size, b[0] + b_size) - max(a[0], b[0]))
    h = max(0, min(a[1] + a_size, b[1] + b_size) - max(a[1], b[1]))
    return w * h


def test_the_crop_finds_the_detail_and_stays_inside_the_frame():
    for x, y in ((60, 60), (820, 540)):
        _, location = preview.detailed_crop(_one_detailed_patch(x, y))
        assert _overlap(location, config.PREVIEW_CROP, (x, y), 300) > 300 * 300 * 0.7

    for x, y in ((0, 0), (900, 0), (0, 600), (900, 600)):
        image = _one_detailed_patch(x, y)
        crop, (cx, cy) = preview.detailed_crop(image)
        assert cx >= 0 and cy >= 0
        assert cx + crop.shape[1] <= image.shape[1] and cy + crop.shape[0] <= image.shape[0]
        assert crop.shape[:2] == (config.PREVIEW_CROP, config.PREVIEW_CROP)


def test_small_and_narrow_frames():
    small = np.full((200, 300, 3), 120, np.uint8)
    crop, location = preview.detailed_crop(small)
    assert location == (0, 0) and crop.shape == small.shape

    crop, location = preview.detailed_crop(_one_detailed_patch(1000, 10, size=(320, 1600)))
    assert crop.shape[:2] == (320, config.PREVIEW_CROP) and location[0] + crop.shape[1] <= 1600


def test_noise_is_measured_on_the_whole_frame_not_the_crop():
    """The detailed crop would make the estimator read detail as noise."""
    rng = np.random.default_rng(1)
    base = np.repeat(np.linspace(60, 190, 1600, dtype=np.float32)[None, :], 1200, axis=0)
    image = cv2.cvtColor(np.clip(base + rng.normal(0, 3.0, base.shape), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image[600:1200, 1000:1600] = rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)

    prepared = preview.prepare(image)

    assert prepared.sigma < noise.sigma_from_gray(cv2.cvtColor(prepared.crop, cv2.COLOR_BGR2GRAY))


def test_the_preview_uses_the_same_tone_correction_as_the_saved_copy():
    """T-6 / C-4: a bright frame with one dark detailed patch. The crop alone
    asks for brightening, the whole frame for darkening; the preview must show
    the whole frame's correction (gamma equal within 0.01)."""
    rng = np.random.default_rng(5)
    image = np.full((1200, 1600, 3), 215, np.uint8)
    image[300:900, 400:1000] = rng.integers(0, 90, (600, 600, 3), dtype=np.uint8)

    prepared = preview.prepare(image)
    shown = preview.render(prepared, 60.0, denoise_percent=0, sharpen_percent=0)
    saved = enhance_image(image, prepared.sigma, 60.0, denoise_percent=0, sharpen_percent=0)

    assert tone.correct(prepared.crop, color_percent=0).gamma < 1.0 < saved.gamma
    assert shown.gamma == pytest.approx(saved.gamma, abs=0.01)
    assert shown.gains == saved.gains
    assert shown.clahe == pytest.approx(saved.clahe) and shown.saturation == pytest.approx(saved.saturation)


def test_render_is_the_real_correction():
    crop = np.random.default_rng(3).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    source = preview.PreviewInput(crop=crop, location=(0, 0), sigma=4.0,
                                  stats=crop, noise_crop=noise.center_crop(crop))

    shown = preview.render(source, 65.0, 150, 200, 120)
    real = enhance_image(crop, 4.0, 65.0, 150, 200, 120)

    assert (shown.denoise_h, shown.sharpen_amount, shown.gains, shown.gamma) == (
        real.denoise_h, real.sharpen_amount, real.gains, real.gamma
    )
    assert np.array_equal(shown.image, real.image)
