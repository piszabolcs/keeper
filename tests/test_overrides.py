# Keeper - manual overrides: the photographer overrules the threshold.
# The threshold is one number for the whole folder; the best frame of a weak
# series may fall below it and still be worth keeping.

from pathlib import Path

import config
from core.sharpness import SharpnessResult, is_kept
from utils.file_ops import move_to_rejects, undo_last_run


def _result(name, score, override=None, folder=None):
    path = (folder / name) if folder is not None else Path(name)
    return SharpnessResult(
        path=path, score=score, laplacian=score, tenengrad=score, brenner=score,
        low_resolution=False, error=None, override=override,
    )


def _write(folder, name):
    path = folder / name
    path.write_bytes(b"not a real jpeg, enough for moving")
    return path


def test_without_override_the_threshold_decides():
    assert is_kept(_result("a.jpg", 90.0), 80) is True
    assert is_kept(_result("a.jpg", 70.0), 80) is False


def test_a_manual_decision_beats_the_threshold_in_both_directions():
    assert is_kept(_result("a.jpg", 86.0, override=True), 90) is True
    assert is_kept(_result("a.jpg", 98.0, override=False), 50) is False


def test_an_unreadable_shot_is_always_kept():
    broken = _result("a.jpg", None)
    broken.override = False
    assert is_kept(broken, 50) is True


def test_moving_the_threshold_keeps_the_manual_decision():
    kept = _result("a.jpg", 40.0, override=True)
    assert all(is_kept(kept, t) for t in (0, 25, 50, 75, 100))


def test_a_manually_kept_shot_does_not_move(tmp_path):
    _write(tmp_path, "keep.jpg")
    _write(tmp_path, "go.jpg")
    results = [
        _result("keep.jpg", 30.0, override=True, folder=tmp_path),
        _result("go.jpg", 30.0, folder=tmp_path),
    ]

    report = move_to_rejects(tmp_path, results, threshold=50)

    assert len(report.moved) == 1
    assert (tmp_path / "keep.jpg").exists()
    assert (tmp_path / config.FOLDER_REJECTS / "go.jpg").exists()


def test_a_manually_rejected_sharp_shot_moves_and_can_be_undone(tmp_path):
    _write(tmp_path, "sharp_but_bad.jpg")
    results = [_result("sharp_but_bad.jpg", 99.0, override=False, folder=tmp_path)]

    assert len(move_to_rejects(tmp_path, results, threshold=50).moved) == 1
    assert (tmp_path / config.FOLDER_REJECTS / "sharp_but_bad.jpg").exists()

    assert len(undo_last_run(tmp_path).restored) == 1
    assert (tmp_path / "sharp_but_bad.jpg").exists()


def test_all_manually_kept_nothing_moves(tmp_path):
    _write(tmp_path, "a.jpg")
    _write(tmp_path, "b.jpg")
    results = [
        _result("a.jpg", 10.0, override=True, folder=tmp_path),
        _result("b.jpg", 10.0, override=True, folder=tmp_path),
    ]

    assert move_to_rejects(tmp_path, results, threshold=90).moved == []
    assert not (tmp_path / config.FOLDER_REJECTS).exists()
