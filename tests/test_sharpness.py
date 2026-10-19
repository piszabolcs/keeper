# Keeper - sharpness metrics and file measurement.

import cv2
import numpy as np
import pytest
from PIL import Image

import config
from core.sharpness import (
    brenner,
    laplacian_variance,
    load_gray_for_measurement,
    measure_sharpness,
    normalize_log,
    tenengrad,
)


def test_normalize_log_clips_and_is_logarithmic():
    assert normalize_log(3.0, 5.0, 600.0) == 0.0
    assert normalize_log(9999.0, 5.0, 600.0) == 100.0
    assert normalize_log(0.0, 5.0, 600.0) == normalize_log(-1.0, 5.0, 600.0) == 0.0
    # the geometric mean sits in the middle on a log scale (8.4 if linear)
    assert 45.0 < normalize_log((5.0 * 600.0) ** 0.5, 5.0, 600.0) < 55.0


def test_a_bad_calibration_raises():
    with pytest.raises(ValueError):
        normalize_log(10.0, 600.0, 5.0)


def checkerboard(size=512, square=64):
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if ((x // square) + (y // square)) % 2 == 0:
                image[y:y + square, x:x + square] = 255
    return image


def test_all_three_metrics_prefer_the_sharp_image():
    sharp = checkerboard()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

    assert laplacian_variance(sharp) > laplacian_variance(blurred)
    assert tenengrad(sharp) > tenengrad(blurred)
    assert brenner(sharp) > brenner(blurred)


def test_flat_and_narrow_images_do_not_raise():
    flat = np.full((256, 256), 128, dtype=np.uint8)
    assert laplacian_variance(flat) == tenengrad(flat) == brenner(flat) == 0.0
    assert brenner(np.full((10, 2), 128, dtype=np.uint8)) == 0.0


def saved_checkerboard(tmp_path, size, name="test.jpg"):
    path = tmp_path / name
    Image.fromarray(checkerboard(size, size // 8)).save(path, quality=95)
    return path


def test_large_images_are_scaled_small_ones_only_flagged(tmp_path):
    gray, low_resolution, _ = load_gray_for_measurement(saved_checkerboard(tmp_path, 2048))
    assert max(gray.shape) == config.WORK_RESOLUTION and low_resolution is False
    assert gray.ndim == 2 and gray.dtype == np.uint8

    gray, low_resolution, _ = load_gray_for_measurement(saved_checkerboard(tmp_path, 512, "s.jpg"))
    assert max(gray.shape) == 512 and low_resolution is True


def test_sharp_beats_blurred_and_parts_are_filled(tmp_path):
    sharp_path = saved_checkerboard(tmp_path, 1024, "sharp.jpg")
    blurred_path = tmp_path / "blurred.jpg"
    Image.fromarray(cv2.GaussianBlur(np.array(Image.open(sharp_path)), (21, 21), 0)).save(
        blurred_path, quality=95
    )

    sharp, blurred = measure_sharpness(sharp_path), measure_sharpness(blurred_path)

    assert sharp.error is None and sharp.score > blurred.score
    for value in (sharp.laplacian, sharp.tenengrad, sharp.brenner, sharp.score):
        assert 0.0 <= value <= 100.0


def test_the_score_does_not_depend_on_the_resolution(tmp_path):
    small = measure_sharpness(saved_checkerboard(tmp_path, 2048, "a.jpg"))
    large = measure_sharpness(saved_checkerboard(tmp_path, 4096, "b.jpg"))

    assert abs(small.score - large.score) < 3.0


def test_missing_and_broken_files_return_an_error(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    for path in (tmp_path / "missing.jpg", broken):
        result = measure_sharpness(path)
        assert result.score is None and result.error is not None


def test_a_broken_primary_falls_back_to_the_other_file_of_the_shot(tmp_path):
    """C-14: a broken JPEG next to a good file of the same shot is still measured."""
    broken = tmp_path / "P.JPG"
    broken.write_bytes(b"truncated jpeg")
    good = saved_checkerboard(tmp_path, 1024, "P.jpeg")

    result = measure_sharpness(broken, [broken, good])

    assert result.score is not None
    assert result.path == good
    assert result.files == [broken, good]


def test_the_pillow_pixel_limit_is_raised_for_panoramas():
    """C-12: frames above Pillow's ~179 MP default are measured, not rejected."""
    assert Image.MAX_IMAGE_PIXELS == config.MAX_IMAGE_PIXELS > 180_000_000
