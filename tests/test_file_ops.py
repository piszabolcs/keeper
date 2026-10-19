# Keeper - moving, the undo log and undo.

import json
import os

import pytest

import config
from core.sharpness import SharpnessResult
from utils import file_ops
from utils.file_ops import (
    collect_images,
    has_undoable_run,
    last_run_summary,
    log_path,
    long_path,
    move_to_rejects,
    read_log,
    undo_last_run,
    write_log,
)


def make_file(folder, name, content=b"content"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(content)
    return path


def result(path, score):
    return SharpnessResult(
        path=path, score=score, laplacian=score, tenengrad=score,
        brenner=score, low_resolution=False, error=None,
    )


def test_collected_shots_are_sorted_by_name(tmp_path):
    for name in ("c.jpg", "a.jpg", "b.jpg"):
        make_file(tmp_path, name)

    assert [i.primary.name for i in collect_images(tmp_path)] == ["a.jpg", "b.jpg", "c.jpg"]


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------


def test_only_shots_below_the_threshold_move(tmp_path):
    weak = make_file(tmp_path, "weak.jpg")
    strong = make_file(tmp_path, "strong.jpg")

    report = move_to_rejects(tmp_path, [result(weak, 20.0), result(strong, 80.0)], 50)

    assert not weak.exists() and strong.exists()
    assert (tmp_path / config.FOLDER_REJECTS / "weak.jpg").exists()
    assert len(report.moved) == 1 and report.failed == []


def test_unreadable_shot_never_moves(tmp_path):
    broken = make_file(tmp_path, "broken.jpg")
    unreadable = SharpnessResult(
        path=broken, score=None, laplacian=None, tenengrad=None, brenner=None,
        low_resolution=False, error="unreadable",
    )

    assert move_to_rejects(tmp_path, [unreadable], 50).moved == []
    assert broken.exists()


def test_name_clash_gets_a_suffix(tmp_path):
    make_file(tmp_path / config.FOLDER_REJECTS, "img.jpg")
    new = make_file(tmp_path, "img.jpg")

    move_to_rejects(tmp_path, [result(new, 10.0)], 50)

    assert (tmp_path / config.FOLDER_REJECTS / "img.jpg").exists()
    assert (tmp_path / config.FOLDER_REJECTS / "img_1.jpg").exists()


def test_the_log_records_the_moves(tmp_path):
    weak = make_file(tmp_path, "weak.jpg")

    move_to_rejects(tmp_path, [result(weak, 20.0)], 50)

    run = read_log(tmp_path)["runs"][0]
    assert run["kind"] == file_ops.KIND_MOVE
    assert run["status"] == file_ops.STATUS_DONE
    assert run["threshold"] == 50
    assert run["moves"][0]["from"] == "weak.jpg"
    assert run["moves"][0]["to"] == config.FOLDER_REJECTS + "/weak.jpg"


def test_nothing_to_move_creates_nothing(tmp_path):
    strong = make_file(tmp_path, "strong.jpg")

    assert move_to_rejects(tmp_path, [result(strong, 90.0)], 50).moved == []
    assert not (tmp_path / config.FOLDER_REJECTS).exists()
    assert not log_path(tmp_path).exists()


def test_damaged_log_moves_nothing(tmp_path):
    # what cannot be logged cannot be undone, so nothing is touched
    log_path(tmp_path).write_text("{ not json", encoding="utf-8")
    weak = make_file(tmp_path, "weak.jpg")

    report = move_to_rejects(tmp_path, [result(weak, 20.0)], 50)

    assert weak.exists() and report.moved == [] and report.error is not None


def test_results_of_another_folder_are_never_moved(tmp_path):
    """T-1: a stale result list (folder switched during a run) moves nothing."""
    folder_a, folder_b = tmp_path / "A", tmp_path / "B"
    from_a = make_file(folder_a, "a.jpg")
    make_file(folder_b, "b.jpg")

    report = move_to_rejects(folder_b, [result(from_a, 10.0)], 50)

    assert report.moved == [] and len(report.failed) == 1
    assert from_a.exists()
    assert not (folder_b / config.FOLDER_REJECTS).exists()
    assert not (folder_a / config.FOLDER_REJECTS).exists()


def test_a_failing_log_write_keeps_the_previous_log(tmp_path, monkeypatch):
    write_log(tmp_path, {"runs": [{"kind": "move", "moves": []}]})
    before = log_path(tmp_path).read_bytes()

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(file_ops.json, "dump", broken_dump)
    with pytest.raises(OSError):
        write_log(tmp_path, {"runs": [{"kind": "move", "moves": [{"x": 1}]}]})

    assert log_path(tmp_path).read_bytes() == before
    assert not (tmp_path / (config.LOG_FILENAME + ".tmp")).exists()


def test_an_interrupted_run_can_still_be_undone(tmp_path, monkeypatch):
    """T-3: the log is written before the moves, so a crash at the 4th file
    leaves an undoable run that brings back the 3 files already moved."""
    files = [make_file(tmp_path, "f%d.jpg" % i) for i in range(6)]
    real_move = file_ops._move_file
    calls = []

    def crash_on_fourth(src, dst):
        calls.append(src)
        if len(calls) == 4:
            raise SystemExit("process killed")
        real_move(src, dst)

    monkeypatch.setattr(file_ops, "_move_file", crash_on_fourth)
    with pytest.raises(SystemExit):
        move_to_rejects(tmp_path, [result(f, 10.0) for f in files], 50)
    monkeypatch.setattr(file_ops, "_move_file", real_move)

    assert sum(not f.exists() for f in files) == 3
    assert has_undoable_run(tmp_path)
    assert last_run_summary(tmp_path)["interrupted"] is True

    report = undo_last_run(tmp_path)

    assert len(report.restored) == 3 and report.skipped == []
    assert all(f.exists() for f in files)
    assert not has_undoable_run(tmp_path)


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


def test_undo_restores_everything_and_cleans_up(tmp_path):
    weak = make_file(tmp_path, "weak.jpg")
    move_to_rejects(tmp_path, [result(weak, 20.0)], 50)

    report = undo_last_run(tmp_path)

    assert weak.exists() and len(report.restored) == 1
    assert read_log(tmp_path)["runs"] == []
    # neither the emptied folder nor an empty log is left behind
    assert not (tmp_path / config.FOLDER_REJECTS).exists()
    assert not log_path(tmp_path).exists()


def test_undo_resolves_the_suffix(tmp_path):
    make_file(tmp_path / config.FOLDER_REJECTS, "img.jpg")
    new = make_file(tmp_path, "img.jpg")
    move_to_rejects(tmp_path, [result(new, 10.0)], 50)

    undo_last_run(tmp_path)

    assert (tmp_path / "img.jpg").exists()
    assert (tmp_path / config.FOLDER_REJECTS / "img.jpg").exists()


def test_undo_never_overwrites_and_keeps_the_entry_for_later(tmp_path):
    """C-7: a skipped entry stays in the log, so the run can be retried."""
    weak = make_file(tmp_path, "weak.jpg")
    other = make_file(tmp_path, "other.jpg")
    move_to_rejects(tmp_path, [result(weak, 20.0), result(other, 20.0)], 50)
    blocker = make_file(tmp_path, "weak.jpg", b"new content")

    report = undo_last_run(tmp_path)

    assert report.restored == [other] and len(report.skipped) == 1
    assert report.remaining == 1
    assert blocker.read_bytes() == b"new content"
    assert has_undoable_run(tmp_path)

    blocker.unlink()
    assert undo_last_run(tmp_path).restored == [weak]
    assert not has_undoable_run(tmp_path)


def test_has_undoable_run(tmp_path):
    assert has_undoable_run(tmp_path) is False
    weak = make_file(tmp_path, "weak.jpg")
    move_to_rejects(tmp_path, [result(weak, 20.0)], 50)
    assert has_undoable_run(tmp_path) is True
    undo_last_run(tmp_path)
    assert has_undoable_run(tmp_path) is False


def test_damaged_log_offers_no_undo(tmp_path):
    log_path(tmp_path).write_text("{ not json", encoding="utf-8")

    assert has_undoable_run(tmp_path) is False
    assert last_run_summary(tmp_path) is None


def test_crafted_log_cannot_move_files_across_the_folder_boundary(tmp_path):
    """S-1 / T-2: every entry pointing outside the folder is ignored."""
    folder = tmp_path / "photos"
    rejects = folder / config.FOLDER_REJECTS
    payload = make_file(rejects, "payload.bat")
    victim = make_file(tmp_path, "victim.docx")
    outside = str(tmp_path / "outside")
    entries = [
        {"from": "../payload.bat", "to": "rejected/payload.bat"},
        {"from": outside + "\\payload.bat", "to": "rejected/payload.bat"},
        {"from": "C:payload.bat", "to": "rejected/payload.bat"},
        {"from": "\\\\server\\share\\payload.bat", "to": "rejected/payload.bat"},
        {"from": "victim.docx", "to": "../victim.docx"},
        {"from": "victim.docx", "to": str(victim)},
        {"from": "victim.docx", "to": "rejected/../../victim.docx"},
        {"from": "victim.docx", "to": "Documents/victim.docx"},
    ]
    log_path(folder).write_text(json.dumps({"runs": [{"moves": entries}]}), encoding="utf-8")

    report = undo_last_run(folder)

    assert report.restored == []
    assert len(report.skipped) == len(entries)
    assert payload.exists() and victim.exists()
    assert sorted(p.name for p in folder.iterdir()) == [config.FOLDER_REJECTS]


def test_a_legacy_hungarian_log_is_still_undoable(tmp_path):
    """Logs written before the English folder names keep working."""
    stored = make_file(tmp_path / "selejt", "a.jpg")
    log_path(tmp_path).write_text(
        json.dumps({"runs": [{"timestamp": "2026-07-28T10:00:00", "kind": "mozgatas",
                              "threshold": 50, "moves": [{"from": "a.jpg", "to": "selejt/a.jpg"}]}]}),
        encoding="utf-8",
    )

    assert last_run_summary(tmp_path)["kind"] == file_ops.KIND_MOVE
    assert last_run_summary(tmp_path)["count"] == 1

    report = undo_last_run(tmp_path)

    assert report.restored == [tmp_path / "a.jpg"] and not stored.exists()
    assert not (tmp_path / "selejt").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH behaviour")
def test_a_target_beyond_max_path_still_moves_and_undoes(tmp_path):
    """C-9: the source fits in 260 characters, rejected/<name> does not."""
    folder = tmp_path
    while len(str(folder)) < 215:
        folder = folder / ("folder_%02d" % len(folder.parts))
    folder.mkdir(parents=True)
    name = "p" * (255 - len(str(folder)) - 5) + ".jpg"
    source = make_file(folder, name)
    assert len(str(source)) == 255

    report = move_to_rejects(folder, [result(source, 10.0)], 50)

    assert report.failed == [] and len(report.moved) == 1
    assert len(str(folder / config.FOLDER_REJECTS / name)) > 260
    assert os.path.exists(long_path(folder / config.FOLDER_REJECTS / name))
    assert len(undo_last_run(folder).restored) == 1
    assert source.exists()
