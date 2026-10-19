# Keeper - shooting data (EXIF) in one place.
#
# Pillow reads JPEG EXIF but cannot open a Sony .ARW at all, and the preview
# JPEG embedded in an ARW lacks aperture and focal length. An .ARW is a plain
# TIFF, though, so a ~60 line IFD walker reads the full EXIF without another
# dependency.

import io
import math
import os
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

import config
from core import raw_loader

# TIFF/EXIF tags used (numbers from the standard)
EXIF_IFD = 0x8769
TAG_MODEL = 0x0110
TAG_ISO = 0x8827
TAG_EXPOSURE = 0x829A
TAG_FNUMBER = 0x829D
TAG_DATETIME_ORIGINAL = 0x9003
TAG_FOCAL = 0x920A
TAG_FOCAL35 = 0xA405
TAG_LENS = 0xA434

# byte size of each TIFF field type
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
# struct codes of the integer types
TYPE_CODE = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i"}

# Real tag values are tiny; a field claiming more than this (or more than the
# file holds) is malformed and skipped instead of allocating gigabytes (C-10).
MAX_FIELD_BYTES = 1 << 16


@dataclass
class ShotInfo:
    """
    Shooting conditions of one frame. Every field may be None: EXIF can be
    lost on export, and manual lenses report no aperture or focal length.
    """

    iso: int = None
    exposure: float = None    # shutter time in seconds
    aperture: float = None    # f-number
    focal: float = None       # focal length in mm as recorded
    focal35: float = None     # 35 mm equivalent focal length
    captured: float = None    # capture time as a unix timestamp
    camera: str = None
    lens: str = None

    @property
    def effective_focal(self):
        """
        Focal length for the shake estimate. The reciprocal rule is stated in
        35 mm terms, so that comes first; the raw value underestimates shake on
        APS-C but beats nothing.
        """
        if self.focal35:
            return float(self.focal35)
        if self.focal:
            return float(self.focal)
        return None


# ---------------------------------------------------------------------------
# minimal TIFF / IFD walker
# ---------------------------------------------------------------------------


def _decode(kind, count, data, order):
    """Python value of the raw bytes of one IFD field."""
    if kind == 2:  # ASCII up to the terminating zero
        return data.split(b"\x00")[0].decode("utf-8", "replace")

    code = TYPE_CODE.get(kind)
    if code:
        values = struct.unpack(order + code * count, data)
        return values[0] if count == 1 else values

    if kind in (5, 10):  # (S)RATIONAL: numerator/denominator pairs
        code = "II" if kind == 5 else "ii"
        values = []
        for i in range(count):
            numerator, denominator = struct.unpack(order + code, data[i * 8:(i + 1) * 8])
            values.append(numerator / denominator if denominator else 0.0)
        return values[0] if count == 1 else tuple(values)

    return None


def _read_ifd(f, offset, order, file_size):
    """
    One IFD: a 16-bit entry count, then 12-byte entries of tag, type, count
    and four bytes that hold the value itself or an offset to it.
    """
    f.seek(offset)
    header = f.read(2)
    if len(header) < 2:
        return {}

    (count,) = struct.unpack(order + "H", header)
    raw = f.read(12 * count)

    entries = {}
    for i in range(count):
        block = raw[i * 12:(i + 1) * 12]
        if len(block) < 12:
            break

        tag, kind, number, value = struct.unpack(order + "HHI4s", block)
        size = TYPE_SIZE.get(kind, 0) * number
        if size <= 0 or size > min(MAX_FIELD_BYTES, file_size):
            continue

        if size <= 4:
            data = value[:size]
        else:
            (where,) = struct.unpack(order + "I", value)
            saved = f.tell()
            f.seek(where)
            data = f.read(size)
            f.seek(saved)
            if len(data) < size:
                continue

        entries[tag] = _decode(kind, number, data, order)

    return entries


def read_tiff_tags(path):
    """IFD0 + EXIF tags of a TIFF-based file (e.g. .ARW); {} for other files."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) < 8 or header[:2] not in (b"II", b"MM"):
            return {}

        order = "<" if header[:2] == b"II" else ">"
        (magic,) = struct.unpack(order + "H", header[2:4])
        if magic != 42:
            return {}

        (ifd0,) = struct.unpack(order + "I", header[4:8])
        tags = _read_ifd(f, ifd0, order, file_size)
        # the EXIF block is a separate IFD that IFD0 points to
        if isinstance(tags.get(EXIF_IFD), int):
            tags.update(_read_ifd(f, tags[EXIF_IFD], order, file_size))

    return tags


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _first(value):
    """Some cameras write single values as lists (ISOSpeedRatings may be many)."""
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _number(value):
    value = _first(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # 0 means "unknown": manual lenses report a 0 mm focal length / f-number
    return number if number > 0 else None


def _text(value):
    if not value:
        return None
    text = str(value).strip()
    # cameras write "----" for an unrecognised (manual) lens
    if not text or set(text) <= {"-"}:
        return None
    return text


def _timestamp(text):
    """EXIF time has a fixed format: "2026:07:28 14:23:05"."""
    if not text:
        return None
    try:
        return datetime.strptime(str(text).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
    except (ValueError, TypeError):
        return None


def _from_tags(tags):
    iso = _number(tags.get(TAG_ISO))
    return ShotInfo(
        iso=int(iso) if iso else None,
        exposure=_number(tags.get(TAG_EXPOSURE)),
        aperture=_number(tags.get(TAG_FNUMBER)),
        focal=_number(tags.get(TAG_FOCAL)),
        focal35=_number(tags.get(TAG_FOCAL35)),
        captured=_timestamp(tags.get(TAG_DATETIME_ORIGINAL)),
        camera=_text(tags.get(TAG_MODEL)),
        lens=_text(tags.get(TAG_LENS)),
    )


def _jpeg_tags(source):
    with Image.open(source) as img:
        exif = img.getexif()
        tags = dict(exif)
        # the useful fields live in the nested EXIF IFD
        tags.update(dict(exif.get_ifd(EXIF_IFD)))
    return tags


def read(path):
    """
    Shooting data of a frame. NEVER raises: on any problem an empty ShotInfo.
    Everything that uses it tolerates missing data.
    """
    path = Path(path)
    try:
        if raw_loader.is_raw(path):
            # the RAW itself first - only it has the focal length
            info = _from_tags(read_tiff_tags(path))
            if info.iso is not None or info.exposure is not None:
                return info
            # unknown RAW variant: fall back to the embedded preview's header
            preview = raw_loader.preview_jpeg_bytes(path)
            if preview is None:
                return info
            return _from_tags(_jpeg_tags(io.BytesIO(preview)))

        return _from_tags(_jpeg_tags(path))
    except Exception:
        return ShotInfo()


def read_iso(path):
    """Only the ISO (used by noise measurement)."""
    return read(path).iso


# ---------------------------------------------------------------------------
# header-only hints
# ---------------------------------------------------------------------------


def shake_margin_stops(shot):
    """
    Stops of margin against the hand-held reciprocal rule (1/focal seconds),
    or None without enough data. +2 = four times faster than needed, 0 = on
    the rule, -1 = half as fast. The rule knows nothing about stabilisers, so
    callers treat the negative side cautiously.
    """
    if shot is None:
        return None
    focal = shot.effective_focal
    if not focal or not shot.exposure:
        return None
    return math.log2((1.0 / focal) / shot.exposure)


def warnings(shot):
    """
    Hints derived from the header. They never reject a frame and never change
    the score; they explain WHY a soft frame may be soft.
    """
    if shot is None:
        return []

    hints = []
    margin = shake_margin_stops(shot)
    if margin is not None and margin <= config.WARN_SHAKE_STOPS:
        hints.append(
            "likely shaken: %s at %d mm" % (_shutter_text(shot.exposure), round(shot.effective_focal))
        )
    if shot.iso and shot.iso >= config.WARN_HIGH_ISO:
        hints.append("likely noisy: ISO %d" % shot.iso)
    return hints


def _shutter_text(seconds):
    """Photographer's notation: 1/250 s or 2.5 s."""
    if not seconds:
        return "?"
    if seconds >= 1.0:
        return "%.1f s" % seconds
    return "1/%d s" % round(1.0 / seconds)


def describe(shot):
    """One-line summary of the shooting data; empty if nothing is known."""
    if shot is None:
        return ""

    parts = []
    if shot.exposure:
        parts.append(_shutter_text(shot.exposure))
    if shot.aperture:
        parts.append(("f/%.1f" % shot.aperture).replace(".0", ""))
    if shot.iso:
        parts.append("ISO %d" % shot.iso)
    if shot.effective_focal:
        parts.append("%d mm" % round(shot.effective_focal))
    return " · ".join(parts)
