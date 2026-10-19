# Keeper - automatic handling of duplicates (surplus series members).
#
# The key rule: a DUPLICATE is not a REJECT. A reject is technically bad, a
# duplicate is a fine frame that is merely surplus; they go to different
# folders because the photographer treats them differently.

import config
from core import series
from core.sharpness import SharpnessResult
from utils.file_ops import destination_folder, move_to_rejects, read_log, undo_last_run


def _result(path, score, phash=0b1010, captured=0.0):
    return SharpnessResult(
        path=path, score=score, laplacian=score, tenengrad=score, brenner=score,
        low_resolution=False, error=None, phash=phash, captured=captured,
    )


def _file(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"img")
    return path


def _series(tmp_path, scores):
    """One series: identical fingerprints, one second apart."""
    results = [
        _result(_file(tmp_path, "k%d.jpg" % i), score, captured=float(i))
        for i, score in enumerate(scores)
    ]
    return results, series.group_results(results)


# ---------------------------------------------------------------------------
# marking
# ---------------------------------------------------------------------------


def test_the_best_stays_the_rest_are_duplicates(tmp_path):
    results, groups = _series(tmp_path, [40.0, 90.0, 60.0])

    assert series.keep_best_n(groups, 1) == (1, 2)
    assert [r.duplicate for r in results] == [True, False, True]


def test_the_best_two_can_be_kept(tmp_path):
    results, groups = _series(tmp_path, [40.0, 90.0, 60.0])

    assert series.keep_best_n(groups, 2) == (2, 1)
    assert [r.duplicate for r in results] == [True, False, False]


def test_marking_is_idempotent_and_reversible(tmp_path):
    results, groups = _series(tmp_path, [40.0, 90.0, 60.0])

    series.keep_best_n(groups, 1)
    series.keep_best_n(groups, 3)
    assert [r.duplicate for r in results] == [False, False, False]

    series.keep_best_n(groups, 1)
    assert series.clear_duplicates(results) == 2
    assert not any(r.duplicate for r in results)


def test_a_shot_outside_any_series_is_not_marked(tmp_path):
    alone = _result(_file(tmp_path, "alone.jpg"), 70.0, phash=0b1111, captured=99.0)
    results, _ = _series(tmp_path, [40.0, 90.0])

    series.keep_best_n(series.group_results(results + [alone]), 1)

    assert alone.duplicate is False


# ---------------------------------------------------------------------------
# where a shot goes
# ---------------------------------------------------------------------------


def test_destination_rules(tmp_path):
    weak = _result(_file(tmp_path, "a.jpg"), 20.0)
    surplus = _result(_file(tmp_path, "b.jpg"), 90.0)
    surplus.duplicate = True
    weak_surplus = _result(_file(tmp_path, "c.jpg"), 20.0)
    weak_surplus.duplicate = True
    protected = _result(_file(tmp_path, "d.jpg"), 20.0)
    protected.duplicate = True
    protected.override = True
    unreadable = SharpnessResult(
        path=tmp_path / "x.jpg", score=None, laplacian=None, tenengrad=None,
        brenner=None, low_resolution=False, error="broken",
    )
    unreadable.duplicate = True

    assert destination_folder(weak, 50) == config.FOLDER_REJECTS
    # a sharp but surplus frame is NOT a reject
    assert destination_folder(surplus, 50) == config.FOLDER_DUPLICATES
    # technically bad outranks surplus
    assert destination_folder(weak_surplus, 50) == config.FOLDER_REJECTS
    # a manual keep beats both
    assert destination_folder(protected, 50) is None
    assert destination_folder(unreadable, 50) is None


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------


def test_one_run_moves_to_both_folders_and_one_undo_restores_both(tmp_path):
    weak = _result(_file(tmp_path, "weak.jpg"), 20.0)
    surplus = _result(_file(tmp_path, "surplus.jpg"), 90.0)
    surplus.duplicate = True
    kept = _result(_file(tmp_path, "kept.jpg"), 95.0)

    report = move_to_rejects(tmp_path, [weak, surplus, kept], threshold=50)

    assert len(report.moved) == 2
    assert (tmp_path / config.FOLDER_REJECTS / "weak.jpg").exists()
    assert (tmp_path / config.FOLDER_DUPLICATES / "surplus.jpg").exists()
    assert (tmp_path / "kept.jpg").exists()
    assert len(read_log(tmp_path)["runs"]) == 1

    undo_last_run(tmp_path)

    assert (tmp_path / "weak.jpg").exists() and (tmp_path / "surplus.jpg").exists()
    assert not (tmp_path / config.FOLDER_REJECTS).exists()
    assert not (tmp_path / config.FOLDER_DUPLICATES).exists()


def test_no_empty_duplicates_folder_and_the_log_names_the_real_folder(tmp_path):
    weak = _result(_file(tmp_path, "weak.jpg"), 20.0)
    move_to_rejects(tmp_path, [weak], threshold=50)
    assert not (tmp_path / config.FOLDER_DUPLICATES).exists()

    surplus = _result(_file(tmp_path, "surplus.jpg"), 90.0)
    surplus.duplicate = True
    move_to_rejects(tmp_path, [surplus], threshold=50)

    run = read_log(tmp_path)["runs"][-1]
    assert run["moves"][0]["to"] == config.FOLDER_DUPLICATES + "/surplus.jpg"


def test_a_raw_jpeg_pair_goes_to_duplicates_together(tmp_path):
    jpeg, raw = _file(tmp_path, "DSC1.jpg"), _file(tmp_path, "DSC1.arw")
    pair = _result(jpeg, 90.0)
    pair.files = [jpeg, raw]
    pair.duplicate = True

    move_to_rejects(tmp_path, [pair], threshold=50)

    assert (tmp_path / config.FOLDER_DUPLICATES / "DSC1.jpg").exists()
    assert (tmp_path / config.FOLDER_DUPLICATES / "DSC1.arw").exists()


def test_the_whole_chain_from_detection_to_moving(tmp_path):
    results, groups = _series(tmp_path, [40.0, 90.0, 60.0, 75.0])

    assert len(groups) == 1
    series.keep_best_n(groups, 1)
    move_to_rejects(tmp_path, results, threshold=30)

    assert (tmp_path / "k1.jpg").exists()
    duplicates = tmp_path / config.FOLDER_DUPLICATES
    assert sorted(f.name for f in duplicates.iterdir()) == ["k0.jpg", "k2.jpg", "k3.jpg"]
    assert not (tmp_path / config.FOLDER_REJECTS).exists()
