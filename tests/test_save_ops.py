# Keeper - saving corrected copies and undoing them.
#
# The rule this file guards above all: a correction NEVER touches the
# original, and its undo never deletes a file the user has changed.

import json
import os
import struct
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image, ImageCms

import config
from core import exif as exif_reader
from core.sharpness import SharpnessResult
from utils.file_ops import (
    KIND_CORRECTION,
    KIND_MOVE,
    last_run_kind,
    log_path,
    move_to_rejects,
    read_log,
    write_log,
)
from utils.save_ops import (
    corrected_folder,
    corrected_name,
    has_undoable_correction,
    quick_save_name,
    save_corrected,
    save_corrected_as,
    save_corrected_batch,
    undo_last_correction,
)

EXIF_ORIENTATION = 0x0112
EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110
EXIF_SUBIFD = 0x8769
EXIF_ISO = 0x8827


def jpeg_file(folder, name, color="gray", size=32):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    Image.new("RGB", (size, size), color).save(path)
    return path


def exif_jpeg(folder, name, orientation=6):
    """A JPEG with real shooting data and an orientation tag."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    exif = Image.Exif()
    exif[EXIF_MAKE] = "SONY"
    exif[EXIF_MODEL] = "ILCE-7M4"
    exif[EXIF_ORIENTATION] = orientation
    exif.get_ifd(EXIF_SUBIFD)[EXIF_ISO] = 6400
    Image.new("RGB", (32, 32), "gray").save(path, exif=exif)
    return path


def bgr_image(color=(10, 20, 30), size=32):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


# ---------------------------------------------------------------------------
# naming and single saves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected",
    [("DSC01234.ARW", "DSC01234.jpg"), ("DSC01234.JPG", "DSC01234.jpg"), ("img.jpeg", "img.jpg")],
)
def test_the_corrected_copy_is_always_jpg(source, expected):
    assert corrected_name(source) == expected


def test_save_goes_to_the_corrected_folder(tmp_path):
    source = jpeg_file(tmp_path, "a.jpg")

    target = save_corrected(tmp_path, source, bgr_image())

    assert target == corrected_folder(tmp_path) / "a.jpg"
    assert corrected_folder(tmp_path) == tmp_path / config.FOLDER_CORRECTED
    assert target.exists()


def test_the_original_bytes_never_change(tmp_path):
    source = jpeg_file(tmp_path, "a.jpg")
    before = source.read_bytes()

    save_corrected(tmp_path, source, bgr_image())

    assert source.read_bytes() == before


def test_a_name_clash_gets_a_suffix_instead_of_overwriting(tmp_path):
    source = jpeg_file(tmp_path, "a.jpg")
    earlier = jpeg_file(corrected_folder(tmp_path), "a.jpg", color="white")
    earlier_bytes = earlier.read_bytes()

    assert save_corrected(tmp_path, source, bgr_image()).name == "a_1.jpg"
    assert earlier.read_bytes() == earlier_bytes


def test_a_raw_source_becomes_a_jpeg(tmp_path):
    source = tmp_path / "DSC01234.ARW"
    source.write_bytes(b"not a real raw, never read")

    target = save_corrected(tmp_path, source, bgr_image())

    assert target.name == "DSC01234.jpg" and target.exists()
    assert source.read_bytes() == b"not a real raw, never read"


def test_colours_are_not_swapped(tmp_path):
    source = jpeg_file(tmp_path, "a.jpg")

    target = save_corrected(tmp_path, source, bgr_image(color=(255, 0, 0)))  # BGR blue

    with Image.open(target) as image:
        r, g, b = image.convert("RGB").getpixel((16, 16))
    assert b > 200 and r < 60


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_exif_is_carried_over_with_normal_orientation(tmp_path):
    source = exif_jpeg(tmp_path, "a.jpg", orientation=6)

    target = save_corrected(tmp_path, source, bgr_image())

    with Image.open(target) as image:
        exif = image.getexif()
        assert exif.get(EXIF_MAKE) == "SONY"
        assert exif.get(EXIF_MODEL) == "ILCE-7M4"
        assert exif.get_ifd(EXIF_SUBIFD).get(EXIF_ISO) == 6400
        # the pixels are already rotated; inheriting 6 would rotate twice
        assert exif.get(EXIF_ORIENTATION) == 1
    with Image.open(source) as image:
        assert image.getexif().get(EXIF_ORIENTATION) == 6


def test_a_source_without_exif_still_saves(tmp_path):
    assert save_corrected(tmp_path, jpeg_file(tmp_path, "a.jpg"), bgr_image()).exists()


def test_unreadable_exif_does_not_block_saving(tmp_path, monkeypatch):
    source = exif_jpeg(tmp_path, "a.jpg")

    def broken(self, *args, **kwargs):
        raise ValueError("unreadable exif")

    monkeypatch.setattr(Image.Exif, "tobytes", broken)

    assert save_corrected(tmp_path, source, bgr_image()).exists()


def test_the_icc_profile_is_preserved(tmp_path):
    """C-11: an Adobe RGB / P3 export must not lose its colour profile."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "with_profile.jpg"
    Image.new("RGB", (32, 32), "gray").save(source, icc_profile=profile)

    target = save_corrected(tmp_path, source, bgr_image())

    with Image.open(target) as image:
        assert image.info.get("icc_profile") == profile


def _fake_arw(tmp_path, iso=6400, exposure=1 / 60.0):
    """A minimal valid TIFF - exactly how an .ARW starts."""
    entries = [(exif_reader.TAG_ISO, 3, iso), (exif_reader.TAG_EXPOSURE, 5, exposure)]
    exif_offset = 8 + 2 + 12 + 4
    data_offset = exif_offset + 2 + 12 * len(entries) + 4

    data = raw_entries = b""
    for tag, kind, value in entries:
        if kind == 5:
            fraction = Fraction(value).limit_denominator(1000000)
            raw_entries += struct.pack("<HHII", tag, 5, 1, data_offset + len(data))
            data += struct.pack("<II", fraction.numerator, fraction.denominator)
        else:
            raw_entries += struct.pack("<HHIHH", tag, 3, 1, int(value), 0)

    content = b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 1)
    content += struct.pack("<HHII", exif_reader.EXIF_IFD, 4, 1, exif_offset) + struct.pack("<I", 0)
    content += struct.pack("<H", len(entries)) + raw_entries + struct.pack("<I", 0) + data

    path = tmp_path / "DSC0001.arw"
    path.write_bytes(content)
    return path


def test_raw_shooting_data_reaches_the_copy_with_normal_orientation(tmp_path):
    target = save_corrected(tmp_path, _fake_arw(tmp_path), np.full((40, 60, 3), 128, np.uint8))

    saved = exif_reader.read(target)
    assert saved.iso == 6400
    assert saved.exposure == pytest.approx(1 / 60.0, rel=1e-3)
    with Image.open(target) as image:
        assert image.getexif().get(EXIF_ORIENTATION) == 1


def test_a_non_tiff_raw_still_saves(tmp_path):
    junk = tmp_path / "DSC0002.arw"
    junk.write_bytes(b"not a tiff header")

    assert save_corrected(tmp_path, junk, np.full((40, 60, 3), 128, np.uint8)).exists()


def test_save_as_writes_the_chosen_file_without_logging(tmp_path):
    """Quick mode: <stem>_keeper.jpg next to the source, metadata kept."""
    source = exif_jpeg(tmp_path, "single.jpg")
    target = quick_save_name(source)

    assert target == tmp_path / ("single%s.jpg" % config.QUICK_SAVE_SUFFIX)
    save_corrected_as(source, bgr_image(), target)

    with Image.open(target) as image:
        assert image.getexif().get(EXIF_MODEL) == "ILCE-7M4"
    assert not log_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------


def test_a_batch_saves_everything_and_logs_checksums(tmp_path):
    a, b = jpeg_file(tmp_path, "a.jpg"), jpeg_file(tmp_path, "b.jpg")

    report = save_corrected_batch(tmp_path, [(a, bgr_image()), (b, bgr_image())])

    assert report.error is None and len(report.saved) == 2 and report.failed == []
    run = read_log(tmp_path)["runs"][-1]
    assert run["kind"] == KIND_CORRECTION and run["status"] == "done"
    assert [e["path"] for e in run["created"]] == [
        "%s/a.jpg" % config.FOLDER_CORRECTED, "%s/b.jpg" % config.FOLDER_CORRECTED,
    ]
    assert all(len(e["sha256"]) == 64 and e["size"] > 0 for e in run["created"])


def test_no_folder_no_files(tmp_path):
    # a FILE named like the output folder blocks creating it
    (tmp_path / config.FOLDER_CORRECTED).write_bytes(b"in the way")

    report = save_corrected_batch(tmp_path, [(jpeg_file(tmp_path, "a.jpg"), bgr_image())])

    assert report.error is not None and report.saved == []
    assert not log_path(tmp_path).exists()


def test_damaged_log_saves_nothing(tmp_path):
    log_path(tmp_path).write_text("{ not json", encoding="utf-8")

    report = save_corrected_batch(tmp_path, [(jpeg_file(tmp_path, "a.jpg"), bgr_image())])

    assert report.error is not None and report.saved == []
    assert not corrected_folder(tmp_path).exists()


def test_one_bad_item_does_not_stop_the_batch(tmp_path):
    a, bad, b = (jpeg_file(tmp_path, n) for n in ("a.jpg", "bad.jpg", "b.jpg"))

    report = save_corrected_batch(tmp_path, [(a, bgr_image()), (bad, None), (b, bgr_image())])

    assert len(report.saved) == 2 and report.failed[0][0] == bad


def test_an_empty_or_failed_batch_leaves_nothing(tmp_path):
    assert save_corrected_batch(tmp_path, []).saved == []
    report = save_corrected_batch(tmp_path, [(jpeg_file(tmp_path, "bad.jpg"), None)])

    assert len(report.failed) == 1
    assert not corrected_folder(tmp_path).exists()
    assert not log_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# undo - the only place where Keeper deletes a file
# ---------------------------------------------------------------------------


def test_undo_removes_the_copies_and_the_log_entry(tmp_path):
    a, b = jpeg_file(tmp_path, "a.jpg"), jpeg_file(tmp_path, "b.jpg")
    save_corrected_batch(tmp_path, [(a, bgr_image()), (b, bgr_image())])

    report = undo_last_correction(tmp_path)

    assert len(report.restored) == 2
    assert not corrected_folder(tmp_path).exists()
    assert read_log(tmp_path)["runs"] == []
    assert a.exists() and b.exists()


def test_an_edited_copy_survives_the_undo(tmp_path):
    """T-4 / C-21: a copy the user changed since saving is never deleted."""
    a, b = jpeg_file(tmp_path, "a.jpg"), jpeg_file(tmp_path, "b.jpg")
    save_corrected_batch(tmp_path, [(a, bgr_image()), (b, bgr_image())])
    edited = corrected_folder(tmp_path) / "a.jpg"
    Image.new("RGB", (32, 32), "red").save(edited)

    report = undo_last_correction(tmp_path)

    assert edited.exists()
    assert report.restored == [corrected_folder(tmp_path) / "b.jpg"]
    assert "changed" in report.skipped[0][1]
    assert not has_undoable_correction(tmp_path)


def test_a_missing_copy_is_skipped(tmp_path):
    save_corrected_batch(tmp_path, [(jpeg_file(tmp_path, "a.jpg"), bgr_image())])
    (corrected_folder(tmp_path) / "a.jpg").unlink()

    report = undo_last_correction(tmp_path)

    assert report.restored == [] and len(report.skipped) == 1
    assert read_log(tmp_path)["runs"] == []


def test_undo_never_deletes_outside_the_corrected_folder(tmp_path):
    """Path traversal in the log: '../x' and 'corrected/../x' are ignored."""
    folder = tmp_path / "work"
    outside = jpeg_file(tmp_path, "important.jpg")
    inside = jpeg_file(folder, "important.jpg")
    corrected_folder(folder).mkdir()
    write_log(folder, {"runs": [{
        "timestamp": "2026-07-27T10:00:00", "kind": "correction",
        "created": ["../important.jpg", "%s/../important.jpg" % config.FOLDER_CORRECTED,
                    {"path": str(outside), "size": 1, "sha256": "0" * 64}],
    }]})

    report = undo_last_correction(folder)

    assert outside.exists() and inside.exists()
    assert report.restored == [] and len(report.skipped) == 3


def test_legacy_entries_are_judged_by_modification_time(tmp_path):
    """Old Hungarian logs have no checksum: delete only what was not edited later."""
    legacy = tmp_path / "javitott"
    untouched, edited = jpeg_file(legacy, "a.jpg"), jpeg_file(legacy, "b.jpg")
    os.utime(untouched, (1_700_000_000, 1_700_000_000))
    os.utime(edited, (1_700_000_100, 1_700_000_100))
    from datetime import datetime

    logged = datetime.fromtimestamp(1_700_000_010).isoformat(timespec="seconds")
    log_path(tmp_path).write_text(json.dumps({"runs": [{
        "timestamp": logged, "kind": "javitas", "created": ["javitott/a.jpg", "javitott/b.jpg"],
    }]}), encoding="utf-8")

    report = undo_last_correction(tmp_path)

    assert report.restored == [untouched] and not untouched.exists()
    assert edited.exists()


def test_without_a_correction_run_undo_does_nothing(tmp_path):
    report = undo_last_correction(tmp_path)

    assert report.restored == [] and report.skipped == []


# ---------------------------------------------------------------------------
# moves and corrections share one log
# ---------------------------------------------------------------------------


def _result(path, score):
    return SharpnessResult(
        path=path, score=score, laplacian=score, tenengrad=score,
        brenner=score, low_resolution=False, error=None,
    )


def test_the_run_kinds_are_told_apart(tmp_path):
    weak = jpeg_file(tmp_path, "weak.jpg")
    move_to_rejects(tmp_path, [_result(weak, 20.0)], 50)
    assert last_run_kind(tmp_path) == KIND_MOVE
    assert has_undoable_correction(tmp_path) is False

    save_corrected_batch(tmp_path, [(jpeg_file(tmp_path, "a.jpg"), bgr_image())])
    assert last_run_kind(tmp_path) == KIND_CORRECTION
    assert has_undoable_correction(tmp_path) is True


def test_undoing_a_correction_leaves_the_move_entry_alone(tmp_path):
    weak = jpeg_file(tmp_path, "weak.jpg")
    move_to_rejects(tmp_path, [_result(weak, 20.0)], 50)
    save_corrected_batch(tmp_path, [(jpeg_file(tmp_path, "a.jpg"), bgr_image())])

    undo_last_correction(tmp_path)

    runs = read_log(tmp_path)["runs"]
    assert len(runs) == 1 and runs[0]["moves"][0]["from"] == "weak.jpg"


def test_damaged_log_offers_no_correction_undo(tmp_path):
    log_path(tmp_path).write_text("{ not json", encoding="utf-8")

    assert has_undoable_correction(tmp_path) is False
