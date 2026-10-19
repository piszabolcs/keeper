# Keeper - RAW support: extension detection, RAW+JPEG pairing, moving pairs
# together and error tolerance. (No real .ARW is used: the tests prove the
# pairing and moving logic, not LibRaw's decoding.)

import os

import pytest
from PIL import Image

import config
from core import raw_loader
from core.sharpness import measure_sharpness
from utils.file_ops import ImageItem, collect_images, move_to_rejects, read_log, undo_last_run


def empty_file(folder, name):
    path = folder / name
    path.write_bytes(b"")
    return path


def jpeg_file(folder, name, size=64):
    path = folder / name
    Image.new("RGB", (size, size), "gray").save(path)
    return path


def measured_pair(tmp_path):
    jpeg_file(tmp_path, "DSC01234.JPG")
    empty_file(tmp_path, "DSC01234.ARW")
    item = collect_images(tmp_path)[0]
    return measure_sharpness(item.primary, item.files)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DSC01234.ARW", True),
        ("DSC01234.arw", True),
        ("DSC01234.JPG", False),
        ("DSC01234.jpeg", False),
        ("notes.txt", False),
        ("no_extension", False),
    ],
)
def test_raw_detection(name, expected):
    assert raw_loader.is_raw(name) is expected


def test_rawpy_is_installed():
    # without it the RAW path would silently disappear
    assert raw_loader.RAWPY_AVAILABLE


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


def test_a_lone_jpeg_and_a_lone_raw_are_single_shots(tmp_path):
    jpeg_file(tmp_path, "a.jpg")
    empty_file(tmp_path, "b.ARW")

    items = collect_images(tmp_path)

    assert [i.primary.name for i in items] == ["a.jpg", "b.ARW"]
    assert not any(i.is_pair for i in items)


def test_same_stem_raw_and_jpeg_are_one_shot_measured_on_the_jpeg(tmp_path):
    jpeg_file(tmp_path, "dsc01234.jpg")
    empty_file(tmp_path, "DSC01234.ARW")

    items = collect_images(tmp_path)

    assert len(items) == 1 and items[0].is_pair and len(items[0].files) == 2
    assert items[0].primary.suffix.lower() == ".jpg"


def test_non_images_and_subfolders_are_skipped(tmp_path):
    jpeg_file(tmp_path, "a.jpg")
    empty_file(tmp_path, "notes.txt")
    empty_file(tmp_path, "clip.mp4")
    (tmp_path / config.FOLDER_REJECTS).mkdir()
    jpeg_file(tmp_path / config.FOLDER_REJECTS, "old.jpg")

    assert [i.primary.name for i in collect_images(tmp_path)] == ["a.jpg"]


def test_a_missing_folder_is_an_empty_list(tmp_path):
    assert collect_images(tmp_path / "missing") == []


def test_an_image_item_can_be_built_by_hand(tmp_path):
    a = jpeg_file(tmp_path, "a.jpg")
    assert ImageItem(primary=a, files=[a, empty_file(tmp_path, "a.ARW")]).is_pair


# ---------------------------------------------------------------------------
# a pair always moves together
# ---------------------------------------------------------------------------


def test_both_members_of_a_pair_move_and_come_back(tmp_path):
    pair = measured_pair(tmp_path)

    report = move_to_rejects(tmp_path, [pair], threshold=101)

    assert report.error is None and len(report.moved) == 2
    rejects = tmp_path / config.FOLDER_REJECTS
    assert (rejects / "DSC01234.JPG").exists() and (rejects / "DSC01234.ARW").exists()
    moves = read_log(tmp_path)["runs"][-1]["moves"]
    assert {m["from"] for m in moves} == {"DSC01234.JPG", "DSC01234.ARW"}

    assert len(undo_last_run(tmp_path).restored) == 2
    assert (tmp_path / "DSC01234.JPG").exists() and (tmp_path / "DSC01234.ARW").exists()
    assert not rejects.exists()


def test_a_failed_pair_rolls_back_completely(tmp_path, monkeypatch):
    pair = measured_pair(tmp_path)
    real_rename = os.rename
    calls = []

    def second_fails(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("the file is locked")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", second_fails)
    report = move_to_rejects(tmp_path, [pair], threshold=101)
    monkeypatch.setattr(os, "rename", real_rename)

    assert report.moved == [] and report.failed
    assert (tmp_path / "DSC01234.JPG").exists() and (tmp_path / "DSC01234.ARW").exists()
    assert read_log(tmp_path)["runs"] == []
    assert not (tmp_path / config.FOLDER_REJECTS).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows file locking")
def test_a_locked_file_leaves_no_stray_copy(tmp_path):
    """C-8: an open JPEG cannot be renamed; nothing may be copied instead."""
    pair = measured_pair(tmp_path)

    with open(tmp_path / "DSC01234.JPG", "rb"):
        report = move_to_rejects(tmp_path, [pair], threshold=101)

    assert report.moved == [] and report.failed
    rejects = tmp_path / config.FOLDER_REJECTS
    assert not rejects.exists() or not any(rejects.iterdir())
    assert (tmp_path / "DSC01234.JPG").exists() and (tmp_path / "DSC01234.ARW").exists()


# ---------------------------------------------------------------------------
# error tolerance
# ---------------------------------------------------------------------------


def test_a_broken_raw_reports_an_error_and_never_moves(tmp_path):
    broken = tmp_path / "broken.ARW"
    broken.write_bytes(b"not a raw file")

    result = measure_sharpness(broken)

    assert result.score is None and result.error is not None
    assert move_to_rejects(tmp_path, [result], threshold=101).moved == []
    assert broken.exists()


def test_a_jpeg_is_its_own_file_list_and_source(tmp_path):
    path = jpeg_file(tmp_path, "a.jpg")

    result = measure_sharpness(path)

    assert result.files == [path]
    assert result.source == raw_loader.SOURCE_JPEG
    assert len(move_to_rejects(tmp_path, [result], threshold=101).moved) == 1
