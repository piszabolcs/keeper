# Keeper - saving corrected copies, and undoing them.
#
# Same ground rule as sorting: originals are never overwritten. A corrected
# image is a NEW file in the corrected/ subfolder (or, in quick mode, a file
# the user picked in a "Save as" dialog). The log helpers live in file_ops.py,
# the single owner of the log format.

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

import config
from core import exif as exif_reader
from utils.file_ops import (
    KIND_CORRECTION,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    UndoReport,
    _now,
    is_plain_name,
    read_log,
    remove_empty_folders,
    run_kind,
    safe_subfolder,
    unique_destination,
    write_log,
)

EXIF_ORIENTATION_TAG = 0x0112
EXIF_ORIENTATION_NORMAL = 1

# Folders an undo may remove corrected copies from (current and legacy name).
CORRECTED_FOLDERS = frozenset(
    [config.FOLDER_CORRECTED]
    + [old for old, new in config.LEGACY_FOLDERS.items() if new == config.FOLDER_CORRECTED]
)

# A legacy log entry has no checksum; its file counts as unchanged if it was
# not modified after the run was logged (plus this much clock slack).
LEGACY_MTIME_SLACK = 2.0


@dataclass
class SaveReport:
    """What was saved and what failed."""

    saved: list = field(default_factory=list)   # (source, target) pairs
    failed: list = field(default_factory=list)  # (source, reason) pairs
    # set when NOTHING was saved (read-only folder, damaged log ...)
    error: str = None


def corrected_folder(folder):
    return Path(folder) / config.FOLDER_CORRECTED


def corrected_name(source_path):
    """
    Corrected copies are always JPEG: a RAW cannot be written back (the
    corrected image is demosaiced RGB, not sensor data), and JPEG sources
    follow the same rule so there is one simple naming rule.
    """
    return Path(source_path).stem + ".jpg"


def quick_save_name(source_path):
    """Default "Save as" target of quick mode: <stem>_keeper.jpg next to the source."""
    source_path = Path(source_path)
    return source_path.with_name(source_path.stem + config.QUICK_SAVE_SUFFIX + ".jpg")


# ---------------------------------------------------------------------------
# metadata carried over from the source
# ---------------------------------------------------------------------------

# From a RAW header only the SHOOTING data is copied (camera, lens, time,
# settings); pixel-related tags describe the raw file and would lie.
RAW_MAIN_TAGS = (exif_reader.TAG_MODEL, 0x010F)
RAW_EXIF_TAGS = (
    exif_reader.TAG_ISO,
    exif_reader.TAG_EXPOSURE,
    exif_reader.TAG_FNUMBER,
    exif_reader.TAG_DATETIME_ORIGINAL,
    exif_reader.TAG_FOCAL,
    exif_reader.TAG_FOCAL35,
    exif_reader.TAG_LENS,
)


def _source_metadata(source_path):
    """
    (EXIF bytes or None, ICC profile or None) for the corrected copy.

    The Orientation tag is reset to 1: the pixels are already rotated, an
    inherited orientation would rotate the copy a second time. The ICC profile
    is kept so Adobe RGB / P3 sources do not shift colour. Any metadata
    problem yields None - the image matters more than its metadata.
    """
    if Path(source_path).suffix.lower() not in config.JPEG_EXTENSIONS:
        # Pillow cannot open RAW files; their tags come from our TIFF walker.
        # rawpy renders sRGB, so there is no profile to carry over.
        return _raw_exif(source_path), None

    exif_bytes = icc = None
    try:
        with Image.open(source_path) as source:
            icc = source.info.get("icc_profile") or None
            if config.CORRECTED_KEEP_EXIF:
                exif = source.getexif()
                if len(exif):
                    exif[EXIF_ORIENTATION_TAG] = EXIF_ORIENTATION_NORMAL
                    # serialise while the file is open: sub-IFDs load lazily
                    exif_bytes = exif.tobytes()
    except Exception:
        pass

    return exif_bytes, icc


def _raw_exif(source_path):
    """An EXIF block built from the RAW header, or None if there is nothing."""
    if not config.CORRECTED_KEEP_EXIF:
        return None
    try:
        tags = exif_reader.read_tiff_tags(source_path)
        if not tags:
            return None

        exif = Image.Exif()
        for tag in RAW_MAIN_TAGS:
            if tags.get(tag) is not None:
                exif[tag] = tags[tag]
        exif[EXIF_ORIENTATION_TAG] = EXIF_ORIENTATION_NORMAL

        details = exif.get_ifd(exif_reader.EXIF_IFD)
        for tag in RAW_EXIF_TAGS:
            if tags.get(tag) is not None:
                details[tag] = tags[tag]

        if not details and len(exif) <= 1:
            return None
        return exif.tobytes()
    except Exception:
        return None


def write_jpeg(source_path, bgr_image, target):
    """
    Writes an OpenCV BGR uint8 image as JPEG with the source's metadata.
    (BGR -> RGB is mandatory: PIL expects RGB, red and blue would swap.)
    """
    image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    exif_bytes, icc = _source_metadata(source_path)

    options = {"quality": config.CORRECTED_JPEG_QUALITY}
    if exif_bytes is not None:
        options["exif"] = exif_bytes
    if icc is not None:
        options["icc_profile"] = icc

    image.save(target, **options)
    return Path(target)


def save_corrected(folder, source_path, bgr_image):
    """Saves one corrected copy into corrected/ and returns its path."""
    target_dir = corrected_folder(folder)
    target_dir.mkdir(exist_ok=True)
    # ask for the name right before writing: the smallest window for a clash
    target = unique_destination(target_dir, corrected_name(source_path))
    return write_jpeg(source_path, bgr_image, target)


def save_corrected_as(source_path, bgr_image, target):
    """Quick mode: writes to the path the user chose. Nothing is logged."""
    return write_jpeg(source_path, bgr_image, Path(target))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_corrected_batch(folder, items):
    """
    Saves a whole batch; items are (source path, BGR image) pairs and may be
    a generator (the enhance worker yields one image at a time, so only one
    full-size image is in memory).

    Each saved file is logged right away together with its size and SHA-256,
    so undo can later tell whether the user has edited the copy since.
    """
    folder = Path(folder)
    report = SaveReport()

    if isinstance(items, (list, tuple)) and not items:
        return report

    # read the log BEFORE writing anything: what cannot be logged cannot be
    # undone either
    try:
        data = read_log(folder)
    except Exception as exc:
        report.error = "The undo log is damaged, nothing was saved: %s" % exc
        return report

    target_dir = corrected_folder(folder)
    try:
        target_dir.mkdir(exist_ok=True)
    except OSError as exc:
        report.error = "The %s folder cannot be created: %s" % (config.FOLDER_CORRECTED, exc)
        return report

    run = None
    for source, image in items:
        source = Path(source)
        try:
            target = save_corrected(folder, source, image)
            entry = {
                "path": "%s/%s" % (config.FOLDER_CORRECTED, target.name),
                "size": target.stat().st_size,
                "sha256": _sha256(target),
            }
        except Exception as exc:
            # one bad image must not stop the batch
            report.failed.append((source, str(exc)))
            continue

        if run is None:
            run = {
                "timestamp": _now(),
                "kind": KIND_CORRECTION,
                "status": STATUS_IN_PROGRESS,
                "created": [],
            }
            data["runs"].append(run)
        run["created"].append(entry)
        report.saved.append((source, target))
        # corrections take seconds each, so logging every file costs nothing
        write_log(folder, data)

    if run is not None:
        run["status"] = STATUS_DONE
        write_log(folder, data)
    else:
        remove_empty_folders(folder, [config.FOLDER_CORRECTED])

    return report


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


def _last_correction_index(data):
    for index in range(len(data["runs"]) - 1, -1, -1):
        if run_kind(data["runs"][index]) == KIND_CORRECTION and "created" in data["runs"][index]:
            return index
    return None


def has_undoable_correction(folder):
    """True if the log has a correction run. False for a damaged log."""
    try:
        return _last_correction_index(read_log(folder)) is not None
    except Exception:
        return False


def _logged_copy(folder, entry):
    """
    (path, size, sha256) of a logged copy, or None if the entry does not point
    at a plain file name inside the corrected folder. Legacy entries are bare
    strings without size and checksum.
    """
    if isinstance(entry, dict):
        text, size, digest = entry.get("path"), entry.get("size"), entry.get("sha256")
    else:
        text, size, digest = entry, None, None
    if not isinstance(text, str):
        return None

    parts = text.split("/")
    if len(parts) != 2 or parts[0] not in CORRECTED_FOLDERS or not is_plain_name(parts[1]):
        return None

    sub = safe_subfolder(folder, parts[0])
    if sub is None:
        return None
    return sub / parts[1], size, digest


def _unchanged(path, size, digest, run):
    """True if the file is still exactly what Keeper wrote."""
    if digest:
        return path.stat().st_size == size and _sha256(path) == digest

    try:
        logged = datetime.fromisoformat(str(run.get("timestamp"))).timestamp()
    except (TypeError, ValueError):
        return False
    return path.stat().st_mtime <= logged + LEGACY_MTIME_SLACK


def undo_last_correction(folder):
    """
    Removes the copies created by the last correction run.

    This is the only place where Keeper deletes a file, and only a file it
    created itself that is still byte-identical to what it wrote: a copy the
    user has edited since is kept and reported. Log entries pointing outside
    the corrected folder are ignored.
    """
    folder = Path(folder)
    data = read_log(folder)
    report = UndoReport()

    index = _last_correction_index(data)
    if index is None:
        return report

    run = data["runs"][index]
    keep = []

    for entry in run.get("created") or []:
        logged = _logged_copy(folder, entry)
        if logged is None:
            report.skipped.append((str(entry), "points outside the corrected folder, ignored"))
            continue
        path, size, digest = logged

        if not path.is_file():
            report.skipped.append((path, "the file no longer exists"))
            continue

        try:
            unchanged = _unchanged(path, size, digest, run)
        except OSError as exc:
            report.skipped.append((path, str(exc)))
            keep.append(entry)
            continue
        if not unchanged:
            report.skipped.append((path, "changed since it was saved, kept"))
            continue

        try:
            path.unlink()
        except OSError as exc:
            report.skipped.append((path, str(exc)))
            keep.append(entry)
            continue

        report.restored.append(path)

    if keep:
        run["created"] = keep
    else:
        data["runs"].pop(index)
    write_log(folder, data)

    remove_empty_folders(folder, sorted(CORRECTED_FOLDERS))
    report.remaining = len(keep)
    return report
