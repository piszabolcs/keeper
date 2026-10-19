# Keeper - brightness and colour correction. Two opposing duties: fix real
# faults (casts, under/over exposure) and leave intentional tones (sunset,
# moody darkness) alone. Most tests guard the second.

import cv2
import numpy as np
import pytest

import config
from core import tone


def _neutral(brightness=120, size=(300, 400)):
    rng = np.random.default_rng(0)
    image = np.full((size[0], size[1], 3), brightness, np.float32) + rng.normal(0, 8, (size[0], size[1], 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def _detailed(size=(300, 400)):
    """Contrasty: hard-edged patches over the whole tonal range."""
    rng = np.random.default_rng(1)
    image = np.full((size[0], size[1], 3), 128, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, size[1] - 20)), int(rng.integers(0, size[0] - 20))
        value = int(rng.choice([0, 20, 60, 200, 235, 255]))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(15, 70)), y + int(rng.integers(15, 70))),
                      (value, value, value), -1)
    return image


def _cast(image, blue=1.0, green=1.0, red=1.0):
    tinted = image.astype(np.float32) * np.array([blue, green, red], np.float32)
    return np.clip(tinted, 0, 255).astype(np.uint8)


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# white balance
# ---------------------------------------------------------------------------


def test_a_neutral_frame_gets_no_correction_and_a_cast_is_found():
    assert tone.limited_gains(tone.white_balance_gains(_neutral())) == (1.0, 1.0, 1.0)

    gains = tone.white_balance_gains(_cast(_neutral(), blue=0.8, red=1.25))
    assert gains[0] > 1.0 and gains[2] < 1.0


def test_correction_moves_towards_neutral():
    image = _cast(_neutral(), blue=0.8, red=1.25)

    def spread(k):
        means = [float(k[:, :, ch].mean()) for ch in range(3)]
        return max(means) - min(means)

    assert spread(tone.correct(image, 100, 100).image) < spread(image)


def test_the_limit_protects_an_intentional_tone():
    sunset = _cast(_neutral(), blue=0.35, green=0.7)
    gains = tone.limited_gains(tone.white_balance_gains(sunset), 300)

    assert max(gains) / min(gains) <= config.WB_GAIN_LIMIT ** 2 + 1e-9
    corrected = tone.correct(sunset, 300, 300).image
    assert float(corrected[:, :, 2].mean()) > float(corrected[:, :, 0].mean())  # still warm


def test_zero_percent_and_black_frames_are_untouched():
    image = _cast(_neutral(), blue=0.8, red=1.25)
    result = tone.correct(image, 0)
    assert result.gains == (1.0, 1.0, 1.0) and result.gamma == pytest.approx(1.0)
    assert np.array_equal(result.image, image)

    black = np.zeros((80, 80, 3), np.uint8)
    assert tone.white_balance_gains(black) == (1.0, 1.0, 1.0)
    assert np.array_equal(tone.correct(black).image, black)


def _green_red_ratio(image):
    grass = image[100:, :]
    return float(grass[:, :, 1].mean()) / float(grass[:, :, 2].mean())


def _grass():
    image = _neutral(brightness=140)
    image[100:, :] = _cast(image[100:, :], blue=0.35, green=1.3, red=0.4)
    return image


def test_a_large_green_subject_is_untouched_by_default():
    """Two thirds of grass: with white balance off by default nothing changes."""
    image = _grass()
    result = tone.correct(image)

    assert result.gains == (1.0, 1.0, 1.0)
    assert _green_red_ratio(result.image) == pytest.approx(_green_red_ratio(image), abs=0.01)


def test_when_enabled_it_still_beats_naive_grey_world():
    image = _grass()
    means = [float(image[:, :, ch].mean()) for ch in range(3)]
    naive = np.clip(image.astype(np.float32) * (sum(means) / 3.0 / np.array(means, np.float32)), 0, 255).astype(np.uint8)

    corrected = tone.correct(image, 100, 100).image

    grass = corrected[100:, :]
    assert float(grass[:, :, 1].mean()) > float(grass[:, :, 0].mean())
    assert float(grass[:, :, 1].mean()) > float(grass[:, :, 2].mean())
    assert _green_red_ratio(corrected) > _green_red_ratio(naive)


def test_white_balance_never_brightens_and_nothing_new_clips():
    for cast in (dict(blue=0.8, red=1.25), dict(blue=1.3, red=0.7), dict(green=1.4)):
        for percent in (50, 100, 300):
            gains = tone.limited_gains(tone.white_balance_gains(_cast(_neutral(200), **cast)), percent)
            assert max(gains) <= 1.0 + 1e-9

    rng = np.random.default_rng(3)
    bright = np.full((300, 400, 3), 210, np.float32)
    bright[40:90, 60:140], bright[200:240, 300:380] = 248, 252
    bright = _cast(np.clip(bright + rng.normal(0, 6, bright.shape), 0, 255).astype(np.uint8), blue=0.9, red=1.12)

    def clipped(image):
        return float(np.mean(image.max(axis=2) >= 250))

    for percent in (50, 100, 300):
        assert clipped(tone.correct(bright, percent, percent).image) <= clipped(bright) + 1e-9


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------


def test_exposure_direction():
    assert tone.exposure_gamma(_gray(_neutral(120))) == pytest.approx(1.0)
    assert tone.exposure_gamma(_gray(_neutral(35))) < 1.0
    assert tone.exposure_gamma(_gray(_neutral(225))) > 1.0
    for value in (0, 255):
        assert tone.exposure_gamma(_gray(np.full((40, 40, 3), value, np.uint8))) == pytest.approx(1.0)


def test_only_up_to_the_edge_of_the_band():
    dark = _neutral(35)
    lifted = tone.apply_gamma(dark, tone.exposure_gamma(_gray(dark)))
    assert float(np.median(_gray(lifted))) == pytest.approx(config.EXPOSURE_TARGET_LOW, abs=6.0)

    result = tone.correct(_neutral(195))
    assert result.gamma > 1.0
    assert float(np.median(_gray(result.image))) == pytest.approx(config.EXPOSURE_TARGET_HIGH, abs=6.0)


def test_gamma_keeps_black_and_white_and_is_limited():
    ramp = np.zeros((10, 256, 3), np.uint8)
    ramp[:, :, :] = np.arange(256, dtype=np.uint8)[None, :, None]
    for gamma in (0.5, 0.8, 1.3, 1.8):
        row = tone.apply_gamma(ramp, gamma)[0, :, 0].astype(int)
        assert row[0] == 0 and row[255] == 255 and all(row[i] <= row[i + 1] for i in range(255))

    assert tone.exposure_gamma(_gray(_neutral(4)), 300) >= 1.0 / config.EXPOSURE_GAMMA_LIMIT - 1e-9
    blinding = tone.correct(_neutral(235))
    assert blinding.gamma == pytest.approx(config.EXPOSURE_GAMMA_LIMIT)
    assert config.EXPOSURE_TARGET_HIGH < float(np.median(_gray(blinding.image))) < 235


def test_a_well_lit_frame_keeps_its_brightness_even_with_a_cast():
    """White balance darkens; the exposure step must give that back."""
    for brightness in (90, 120, 150):
        image = _cast(_neutral(brightness), blue=0.85, red=1.2)
        before = float(np.median(_gray(image)))
        after = float(np.median(_gray(tone.correct(image, 100, 100).image)))
        assert after == pytest.approx(before, abs=3.0), brightness


def test_measure_and_both_steps_on_a_dark_tinted_frame():
    image = _cast(_neutral(45), blue=0.85, red=1.2)
    gains, median = tone.measure(image)
    assert tone.wb_deviation(gains) > config.WB_MIN_DEVIATION and median < config.EXPOSURE_TARGET_LOW

    result = tone.correct(_cast(_neutral(40), blue=0.8, red=1.25), 100, 100)
    assert result.gains != (1.0, 1.0, 1.0) and result.gamma < 1.0
    assert float(np.median(_gray(result.image))) > 40


# ---------------------------------------------------------------------------
# local contrast and saturation
# ---------------------------------------------------------------------------


def _hazy(image, amount):
    return np.clip(image.astype(np.float32) * (1.0 - amount) + 128.0 * amount, 0, 255).astype(np.uint8)


def _colourful(saturation=200, value=140):
    hsv = np.zeros((120, 240, 3), np.uint8)
    for i in range(8):
        hsv[:, i * 30:(i + 1) * 30] = (i * 22, saturation, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_clahe_only_on_flat_clean_frames():
    contrasty = _detailed()
    assert tone.contrast_span(contrasty) >= config.CONTRAST_TARGET_SPAN
    assert tone.clahe_clip(contrasty) == 0.0

    hazy = _hazy(_detailed(), 0.7)
    clip = tone.clahe_clip(hazy, sigma=0.0)
    assert clip > 0.0 and tone.contrast_span(tone.apply_clahe(hazy, clip)) > tone.contrast_span(hazy)
    # noise brake: a noisy hazy frame is not stretched
    assert tone.clahe_clip(hazy, sigma=6.0) == 0.0
    assert tone.clahe_clip(np.zeros((80, 80, 3), np.uint8)) == 0.0


def test_saturation_lifts_only_dull_frames():
    assert tone.saturation_scale(_colourful(200)) == 1.0
    assert 1.0 < tone.saturation_scale(_colourful(40)) <= config.SATURATION_SCALE_LIMIT


def test_the_hard_limits_hold_at_any_slider_position_and_the_slider_still_scales():
    hazy = _hazy(_detailed(), 0.7)
    for percent in (0, 50, 100, 200, config.ENHANCE_SCALE_MAX):
        assert tone.clahe_clip(hazy, percent, sigma=0.0) <= config.CLAHE_CLIP_MAX + 1e-9
        for saturation in (10, 40, 70):
            assert tone.saturation_scale(_colourful(saturation), percent) <= config.SATURATION_SCALE_LIMIT + 1e-9

    dull = _colourful(10)
    assert 1.0 < tone.saturation_scale(dull, 50) < tone.saturation_scale(dull, 100)
    assert 0.0 < tone.clahe_clip(hazy, 50, sigma=0.0) < tone.clahe_clip(hazy, 100, sigma=0.0)


def test_saturation_never_clips_and_ignores_unusable_pixels():
    for value in (200, 254, 255):
        image = _colourful(40, value)
        boosted = tone.apply_saturation(image, config.SATURATION_SCALE_LIMIT)
        assert np.array_equal(boosted.max(axis=2), image.max(axis=2))

    image = _colourful(180)
    spoiled = image.copy()
    spoiled[:60], spoiled[60:90] = 0, 255
    assert tone.saturation_level(spoiled) == pytest.approx(tone.saturation_level(image), abs=6.0)


def test_zero_colour_percent_does_nothing():
    hazy = _hazy(_detailed(), 0.7)
    image, clip, scale = tone.colorize(hazy, 0)
    assert clip == 0.0 and scale == pytest.approx(1.0) and np.array_equal(image, hazy)


def test_statistics_come_from_a_bounded_copy_of_the_whole_frame():
    large = cv2.resize(_detailed(), (2400, 1800), interpolation=cv2.INTER_NEAREST)
    stats = tone.stats_image(large)

    assert max(stats.shape[:2]) == config.TONE_STATS_RESOLUTION
    assert tone.stats_image(_detailed()) is not None and tone.stats_image(_neutral()).shape == (300, 400, 3)
