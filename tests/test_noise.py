# Keeper - noise estimation and the ISO sanity cap.

import numpy as np
import pytest
from PIL import Image

import config
from core import noise
from core.noise import estimate_sigma, measure_noise, read_iso

SEED = 20260727


def flat(value=128, size=512):
    return np.full((size, size), value, dtype=np.uint8)


def noisy(sigma, value=128, size=512):
    rng = np.random.default_rng(SEED)
    image = np.full((size, size), float(value)) + rng.normal(0.0, sigma, (size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


def texture(size=512, step=2):
    """Dense checkerboard: all high-frequency detail, no noise."""
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, step):
        for x in range(0, size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                image[y:y + step, x:x + step] = 255
    return image


def test_flat_is_almost_zero_and_known_noise_is_recovered():
    assert estimate_sigma(flat()) < 1.0
    assert 3.5 < estimate_sigma(noisy(5.0)) < 6.5
    assert estimate_sigma(noisy(8.0)) > estimate_sigma(noisy(2.0))
    assert estimate_sigma(np.zeros((2, 2), dtype=np.uint8)) >= 0.0


def test_detail_is_not_mistaken_for_noise():
    """Blocks + low percentile: a half-textured clean frame stays below a noisy flat one."""
    half_textured = flat()
    half_textured[:, :256] = texture()[:, :256]

    assert estimate_sigma(half_textured) < estimate_sigma(noisy(6.0))


def test_a_fully_textured_frame_is_a_known_limitation():
    assert estimate_sigma(texture()) > 10.0


def saved(tmp_path, array, name="test.jpg"):
    path = tmp_path / name
    # quality 100, 4:4:4: JPEG compression itself smooths the noise away
    Image.fromarray(array).save(path, quality=100, subsampling=0)
    return path


def test_files_are_measured_at_native_resolution(tmp_path):
    clean = saved(tmp_path, flat(), "clean.jpg")
    large_noisy = saved(tmp_path, noisy(8.0, size=2400), "noisy.jpg")

    assert measure_noise(large_noisy).sigma > 4.0 > measure_noise(clean).sigma


def test_missing_or_broken_file_has_no_sigma_but_an_error(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")

    for path in (tmp_path / "missing.jpg", broken):
        result = measure_noise(path)
        assert result.sigma is None and result.error


def test_iso_is_read_when_present(tmp_path):
    path = saved(tmp_path, noisy(3.0), "iso.jpg")
    assert read_iso(path) is None
    assert read_iso(tmp_path / "missing.jpg") is None

    with Image.open(path) as image:
        exif = image.getexif()
        exif.get_ifd(0x8769)[0x8827] = 3200
        image.save(path, quality=100, subsampling=0, exif=exif)

    assert read_iso(path) == 3200
    assert measure_noise(path).iso == 3200


def test_low_iso_caps_implausible_noise_but_keeps_plausible_values():
    # one real ISO 100 frame measured 4.41 because of fine detail
    assert noise.iso_capped_sigma(4.41, 100) == pytest.approx(config.ISO_SIGMA_CAP_BASE)
    for sigma in (0.46, 0.57, 1.41):
        assert noise.iso_capped_sigma(sigma, 100) == pytest.approx(sigma)


def test_the_cap_grows_with_the_square_root_of_iso():
    assert noise.iso_capped_sigma(9.0, 6400) == pytest.approx(9.0)
    assert noise.iso_capped_sigma(99.0, 400) == pytest.approx(2.0 * noise.iso_capped_sigma(99.0, 100))


def test_our_own_brightening_relaxes_the_cap_but_darkening_does_not_tighten_it():
    assert noise.iso_capped_sigma(6.0, 100, brightening=3.0) == pytest.approx(
        config.ISO_SIGMA_CAP_BASE * 3.0
    )
    assert noise.iso_capped_sigma(6.0, 100, brightening=0.4) == pytest.approx(
        config.ISO_SIGMA_CAP_BASE
    )


def test_without_iso_there_is_no_cap():
    assert noise.iso_capped_sigma(9.9, None) == pytest.approx(9.9)
    assert noise.iso_capped_sigma(9.9, 0) == pytest.approx(9.9)
    assert noise.iso_capped_sigma(None, 100) is None
