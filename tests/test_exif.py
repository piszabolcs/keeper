# Keeper - reading shooting data: Pillow for JPEG, our TIFF walker for RAW.
# Missing data matters as much as present data.

import struct
import time
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image

import config
from core import exif


def _jpeg(tmp_path, tags, name="p.jpg"):
    image = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
    data = image.getexif()
    details = data.get_ifd(exif.EXIF_IFD)
    for tag, value in tags.items():
        details[tag] = value
    path = tmp_path / name
    image.save(path, exif=data)
    return path


def test_jpeg_shooting_data_is_read(tmp_path):
    path = _jpeg(tmp_path, {
        exif.TAG_ISO: 400, exif.TAG_EXPOSURE: 1 / 250.0, exif.TAG_FNUMBER: 2.8,
        exif.TAG_FOCAL: 85.0, exif.TAG_FOCAL35: 85,
        exif.TAG_DATETIME_ORIGINAL: "2026:07:28 14:23:05",
    })

    info = exif.read(path)

    assert info.iso == 400
    assert info.exposure == pytest.approx(1 / 250.0, rel=1e-3)
    assert info.aperture == pytest.approx(2.8, rel=1e-3)
    assert info.effective_focal == pytest.approx(85.0)
    assert info.captured == pytest.approx(time.mktime((2026, 7, 28, 14, 23, 5, 0, 0, -1)))


def test_missing_broken_or_exifless_files_give_empty_data(tmp_path):
    bare = tmp_path / "bare.jpg"
    Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(bare)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    for path in (bare, broken, tmp_path / "missing.jpg"):
        info = exif.read(path)
        assert info.iso is None and info.exposure is None and info.captured is None
        assert exif.describe(info) == ""


def test_zero_focal_and_aperture_mean_unknown(tmp_path):
    """Manual lenses record 0 - that is a missing value, not a number."""
    info = exif.read(_jpeg(tmp_path, {exif.TAG_FNUMBER: 0.0, exif.TAG_FOCAL: 0.0}))
    assert info.aperture is None and info.focal is None and info.effective_focal is None


def test_list_iso_and_the_35mm_equivalent(tmp_path):
    assert exif.read(_jpeg(tmp_path, {exif.TAG_ISO: (800, 0, 0)}, "a.jpg")).iso == 800
    info = exif.read(_jpeg(tmp_path, {exif.TAG_FOCAL: 35.0, exif.TAG_FOCAL35: 52}, "b.jpg"))
    assert info.effective_focal == pytest.approx(52.0)


def _tiff(tmp_path, tags, name="p.arw"):
    """A minimal little-endian TIFF - exactly how an .ARW is structured."""
    entries = sorted(tags.items())
    exif_offset = 8 + 2 + 12 + 4
    data_offset = exif_offset + 2 + 12 * len(entries) + 4

    data = raw = b""
    for tag, (kind, value) in entries:
        if kind == 5:  # RATIONAL - does not fit into four bytes
            fraction = Fraction(value).limit_denominator(1000000)
            raw += struct.pack("<HHII", tag, 5, 1, data_offset + len(data))
            data += struct.pack("<II", fraction.numerator, fraction.denominator)
        elif kind == "huge":  # a field claiming far more bytes than exist
            raw += struct.pack("<HHII", tag, 5, value, data_offset)
        else:
            raw += struct.pack("<HHIHH", tag, 3, 1, int(value), 0)

    content = b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 1)
    content += struct.pack("<HHII", exif.EXIF_IFD, 4, 1, exif_offset) + struct.pack("<I", 0)
    content += struct.pack("<H", len(entries)) + raw + struct.pack("<I", 0) + data

    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_the_tiff_walker_reads_the_tags(tmp_path):
    path = _tiff(tmp_path, {
        exif.TAG_ISO: (3, 6400), exif.TAG_EXPOSURE: (5, 1 / 60.0), exif.TAG_FOCAL: (5, 50.0),
    })

    tags = exif.read_tiff_tags(path)

    assert tags[exif.TAG_ISO] == 6400
    assert tags[exif.TAG_EXPOSURE] == pytest.approx(1 / 60.0, rel=1e-3)
    assert tags[exif.TAG_FOCAL] == pytest.approx(50.0, rel=1e-3)


def test_non_tiff_and_truncated_files_give_no_tags(tmp_path):
    other = tmp_path / "other.bin"
    other.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    truncated = tmp_path / "truncated.arw"
    truncated.write_bytes(b"II" + struct.pack("<HI", 42, 9999))

    assert exif.read_tiff_tags(other) == {}
    assert exif.read_tiff_tags(truncated) == {}


def test_a_forged_field_size_is_skipped_without_allocating(tmp_path):
    """C-10: a field claiming 400 million RATIONALs must not allocate gigabytes."""
    path = _tiff(tmp_path, {exif.TAG_ISO: (3, 800), exif.TAG_EXPOSURE: ("huge", 400_000_000)})

    tags = exif.read_tiff_tags(path)

    assert tags[exif.TAG_ISO] == 800
    assert exif.TAG_EXPOSURE not in tags


class _Shot:
    def __init__(self, iso=None, exposure=None, focal=None, aperture=None):
        self.iso, self.exposure, self.aperture, self.focal35 = iso, exposure, aperture, focal

    @property
    def effective_focal(self):
        return self.focal35


def test_the_reciprocal_rule_margin():
    assert exif.shake_margin_stops(_Shot(exposure=1 / 50.0, focal=50)) == pytest.approx(0.0)
    assert exif.shake_margin_stops(_Shot(exposure=1 / 200.0, focal=50)) == pytest.approx(2.0)
    assert exif.shake_margin_stops(_Shot(exposure=1 / 25.0, focal=50)) == pytest.approx(-1.0)
    assert exif.shake_margin_stops(_Shot(exposure=1 / 50.0)) is None
    assert exif.shake_margin_stops(None) is None


def test_warnings_for_shake_and_high_iso_but_no_guessing():
    assert any("shaken" in w for w in exif.warnings(_Shot(iso=100, exposure=1 / 15.0, focal=200)))
    assert any("noisy" in w for w in exif.warnings(_Shot(iso=config.WARN_HIGH_ISO, exposure=1 / 500.0)))
    assert exif.warnings(_Shot(iso=100, exposure=1 / 500.0, focal=35)) == []
    # without a focal length nothing is known about shake
    assert exif.warnings(_Shot(iso=100, exposure=1.0)) == []


def test_describe_uses_photographer_notation():
    text = exif.describe(_Shot(iso=100, exposure=1 / 250.0, focal=35, aperture=2.8))

    assert text == "1/250 s · f/2.8 · ISO 100 · 35 mm"
    assert exif.describe(_Shot(exposure=2.5)) == "2.5 s"
    assert exif.describe(_Shot(iso=800)) == "ISO 800"
