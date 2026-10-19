# Keeper - reading RAW files.
#
# RAW files are only ever READ, never written or converted. For measuring we
# use the JPEG preview the camera embeds in every RAW: it is the same picture
# with the camera's own rendering, and softness survives JPEG compression, so
# it is as good for focus checks as a full demosaic - and orders of magnitude
# faster (a 60 MP demosaic takes seconds per file).

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import config

# rawpy is optional: without it JPEG mode keeps working and RAW files report
# an error instead of crashing the program at start-up.
try:
    import rawpy

    RAWPY_AVAILABLE = True
except Exception:  # pragma: no cover - always installed in the project venv
    rawpy = None
    RAWPY_AVAILABLE = False

# Where the measured pixels came from. The camera preview is sharpened, a
# demosaiced image is not, so their scores are not directly comparable.
SOURCE_JPEG = "jpeg"
SOURCE_RAW_PREVIEW = "raw-preview"
SOURCE_RAW_DEMOSAIC = "raw-demosaic"

_MISSING_RAWPY = "reading RAW files needs the rawpy package: pip install rawpy"


def is_raw(path):
    """True if the extension says RAW."""
    return Path(path).suffix.lower() in config.RAW_EXTENSIONS


def _gray_from_preview(thumb):
    """The embedded preview as a grey array, or None for unknown formats."""
    if thumb.format == rawpy.ThumbFormat.JPEG:
        with Image.open(io.BytesIO(thumb.data)) as img:
            # libjpeg scales while decoding (DCT level) - same trick as JPEGs
            img.draft("L", (config.WORK_RESOLUTION, config.WORK_RESOLUTION))
            # portrait previews carry an EXIF rotation; Brenner is directional
            img = ImageOps.exif_transpose(img)
            return np.array(img.convert("L"))

    if thumb.format == rawpy.ThumbFormat.BITMAP:
        # may point into rawpy's memory: copy before the file is closed
        data = np.array(thumb.data, copy=True)
        return cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) if data.ndim == 3 else data

    return None


def load_gray_from_raw(path):
    """
    (grey array, source) of a RAW file. Not scaled to the work resolution:
    the caller does that, so RAW and JPEG go through the same normalisation.
    """
    if not RAWPY_AVAILABLE:
        raise RuntimeError(_MISSING_RAWPY)

    with rawpy.imread(str(path)) as raw:
        try:
            gray = _gray_from_preview(raw.extract_thumb())
        except Exception:
            gray = None  # no (readable) preview: take the slow path

        # a tiny thumbnail would have to be upscaled - misleading, skip it
        if gray is not None and max(gray.shape[:2]) >= config.WORK_RESOLUTION:
            return gray, SOURCE_RAW_PREVIEW

        # half-size demosaic: 4x faster and still above the work resolution
        rgb = raw.postprocess(
            half_size=True, use_camera_wb=True, no_auto_bright=True, output_bps=8
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), SOURCE_RAW_DEMOSAIC


def preview_jpeg_bytes(path):
    """The embedded JPEG preview, or None. Used for thumbnails."""
    if not RAWPY_AVAILABLE:
        return None
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                return bytes(thumb.data)
    except Exception:
        return None
    return None


def demosaic_rgb(path):
    """Half-size demosaiced RGB - for measuring/thumbnails without a preview."""
    if not RAWPY_AVAILABLE:
        raise RuntimeError(_MISSING_RAWPY)
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            half_size=True, use_camera_wb=True, no_auto_bright=True, output_bps=8
        )


def demosaic_rgb_full(path):
    """
    FULL resolution demosaiced RGB, for corrected copies: whoever shoots RAW
    wants the full quality back, not a half-size image or the camera JPEG.
    rawpy applies the camera's orientation flag, so the result is upright.
    """
    if not RAWPY_AVAILABLE:
        raise RuntimeError(_MISSING_RAWPY)
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
