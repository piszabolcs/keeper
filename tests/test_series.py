# Keeper - series detection. Grouping is tested with hand-made hash/time
# pairs, so a test fails for the logic, not for a synthetic image that
# happens to look different than expected.

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config
from core import exif as core_exif
from core import series
from core.sharpness import is_kept, measure_sharpness


class Subject:
    """Just the SharpnessResult fields grouping uses."""

    def __init__(self, name, phash, captured, score=50.0):
        self.path = Path(name)
        self.phash = phash
        self.captured = captured
        self.score = score
        self.override = None


def _scene(seed, size=(480, 640)):
    """Photo-like synthetic scene: large soft blobs plus fine texture."""
    rng = np.random.default_rng(seed)
    image = np.zeros(size, np.float32)
    for _ in range(8):
        centre = (int(rng.integers(0, size[1])), int(rng.integers(0, size[0])))
        cv2.circle(image, centre, int(rng.integers(60, 180)), float(rng.integers(40, 220)), -1)
    image = cv2.GaussianBlur(image, (0, 0), 18) + rng.normal(0, 5, size)
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# perceptual hash
# ---------------------------------------------------------------------------


def test_the_hash_is_64_bit_and_deterministic():
    image = _scene(2)
    assert 0 <= series.phash(image) < 2 ** series.PHASH_BITS
    assert series.phash(image) == series.phash(image.copy())


def test_the_hash_survives_brightness_noise_and_blur():
    image = _scene(3)
    brighter = np.clip(image.astype(np.float32) * 1.41, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(0)
    noisier = np.clip(image + rng.normal(0, 12, image.shape), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (0, 0), 6.0)

    for variant in (brighter, noisier, blurred):
        assert series.hamming(series.phash(image), series.phash(variant)) <= 8


def test_different_scenes_are_far_apart():
    hashes = [series.phash(_scene(100 + i)) for i in range(6)]
    distances = [series.hamming(a, b) for i, a in enumerate(hashes) for b in hashes[i + 1:]]

    assert min(distances) > config.SERIES_MAX_DISTANCE
    assert series.hamming(None, 0) == series.hamming(0, None) == series.PHASH_BITS


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def test_close_and_similar_shots_form_a_series():
    found = series.group_results([
        Subject("a.jpg", 0b1010, 1000.0), Subject("b.jpg", 0b1011, 1001.0),
        Subject("c.jpg", 0b1010, 1002.0),
    ])
    assert len(found) == 1 and found[0].size == 3


def test_far_apart_different_or_lonely_shots_do_not():
    assert series.group_results([Subject("a.jpg", 0b1010, 1000.0), Subject("b.jpg", 0b1010, 2800.0)]) == []
    assert series.group_results([
        Subject("a.jpg", 0, 1000.0), Subject("b.jpg", (1 << series.PHASH_BITS) - 1, 1001.0),
    ]) == []
    assert series.group_results([Subject("a.jpg", 0b1010, 1000.0)]) == []


def test_shots_without_hash_or_capture_time_are_left_out():
    found = series.group_results([
        Subject("broken.jpg", None, 1000.0, score=None),
        Subject("no_exif.jpg", 0b1010, None),
        Subject("a.jpg", 0b1010, 1001.0), Subject("b.jpg", 0b1010, 1002.0),
    ])
    assert [m.path.name for m in found[0].members] == ["a.jpg", "b.jpg"]


def test_identical_tripod_shots_without_exif_time_are_no_series(tmp_path):
    """C-13: file times change on copying, so they never make a series."""
    image = _scene(9, size=(1100, 1400))
    for name in ("a.jpg", "b.jpg"):
        Image.fromarray(image).save(tmp_path / name, quality=95)

    results = [measure_sharpness(tmp_path / n) for n in ("a.jpg", "b.jpg")]

    assert all(r.captured is None for r in results)
    assert series.group_results(results) == []


def test_the_capture_time_comes_from_exif(tmp_path):
    path = tmp_path / "with_exif.jpg"
    image = Image.fromarray(_scene(6))
    data = image.getexif()
    data.get_ifd(core_exif.EXIF_IFD)[core_exif.TAG_DATETIME_ORIGINAL] = "2026:07:28 14:23:05"
    image.save(path, exif=data)

    assert measure_sharpness(path).captured == core_exif.read(path).captured is not None


def test_the_group_does_not_drift():
    """Similarity is measured against the FIRST member, not the previous one."""
    found = series.group_results([
        Subject("a.jpg", 0b00000000000000000000, 1000.0),
        Subject("b.jpg", 0b00000000000000011111, 1001.0),
        Subject("c.jpg", 0b00000000001111111111, 1002.0),
        Subject("d.jpg", 0b00000111111111111111, 1003.0),
        Subject("e.jpg", 0b11111111111111111111, 1004.0),
    ])
    assert all(s.size < 5 for s in found)


def test_a_long_burst_stays_together():
    """The time gap is measured against the PREVIOUS member."""
    found = series.group_results([Subject("k%02d.jpg" % i, 0b1010, 1000.0 + 2.0 * i) for i in range(10)])
    assert len(found) == 1 and found[0].size == 10


def test_the_best_is_the_sharpest_readable_member_and_numbering_starts_at_one():
    found = series.group_results([
        Subject("broken.jpg", 0b1010, 999.0, score=None),
        Subject("a.jpg", 0b1010, 1000.0, score=41.0), Subject("b.jpg", 0b1010, 1001.0, score=86.0),
        Subject("c.jpg", (1 << series.PHASH_BITS) - 1, 5000.0), Subject("d.jpg", (1 << series.PHASH_BITS) - 1, 5001.0),
    ])
    assert found[0].best.path.name == "b.jpg"
    assert [s.index for s in found] == [1, 2]


def test_the_info_map_marks_every_member():
    members = [Subject("a.jpg", 0b1010, 1000.0, score=40.0), Subject("b.jpg", 0b1010, 1001.0, score=90.0)]
    lonely = Subject("lonely.jpg", (1 << series.PHASH_BITS) - 1, 9000.0)

    mapping = series.info_map(series.group_results(members + [lonely]))

    assert mapping[Path("a.jpg")].position == 1 and mapping[Path("a.jpg")].size == 2
    assert mapping[Path("a.jpg")].is_best is False and mapping[Path("b.jpg")].is_best is True
    assert Path("lonely.jpg") not in mapping


def test_protect_best_marks_only_the_best_and_is_kept_agrees():
    """The bulk action sets the same override a click would set."""
    members = [
        Subject("a.jpg", 0b1010, 1000.0, score=40.0), Subject("b.jpg", 0b1010, 1001.0, score=30.0),
    ]

    assert series.protect_best(series.group_results(members)) == 1
    assert members[0].override is True and members[1].override is None
    assert is_kept(members[0], 90) is True and is_kept(members[1], 90) is False
