# Keeper - main window workflow (Qt offscreen; no window ever appears).

import logging
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QEvent, QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

import config  # noqa: E402
from core.sharpness import SharpnessResult  # noqa: E402
from gui import worker as worker_module  # noqa: E402
from gui.enhance_worker import EnhanceProgress, EnhanceWorker  # noqa: E402
from gui.main_window import STATE_ENHANCING, STATE_MEASURED, STATE_MEASURING, MainWindow  # noqa: E402
from gui.preview_dialog import PreviewDialog  # noqa: E402
from gui.signature import AboutDialog, SignatureButton  # noqa: E402
from utils.file_ops import has_undoable_run, move_to_rejects  # noqa: E402
from version import __version__  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    w = MainWindow()
    w.folder = tmp_path
    yield w
    w.close()


def _result(folder, name, score):
    path = folder / name
    path.write_bytes(b"not a real jpeg, but it exists")
    return SharpnessResult(
        path=path, score=score, laplacian=score, tenengrad=score, brenner=score,
        low_resolution=False, error=None, files=[path],
    )


def _measured(window, scores):
    """Puts the window into the state right after a finished measurement."""
    window.results = [_result(window.folder, "img%d.jpg" % i, s) for i, s in enumerate(scores)]
    for result in window.results:
        window._add_card(result)
    window.state = STATE_MEASURED
    window._refresh_state()


def _answer(monkeypatch, yes=True):
    button = QMessageBox.StandardButton.Yes if yes else QMessageBox.StandardButton.No
    monkeypatch.setattr("gui.main_window.QMessageBox.question", lambda *a, **k: button)
    monkeypatch.setattr("gui.main_window.QMessageBox.information", lambda *a, **k: None)


def _wait_until(condition, timeout=10.0):
    """Waits with a real event loop (QTest.qWait starves the worker threads)."""
    loop, timer = QEventLoop(), QTimer()
    deadline = time.monotonic() + timeout
    timer.timeout.connect(lambda: (condition() or time.monotonic() > deadline) and loop.quit())
    timer.start(20)
    loop.exec()
    timer.stop()
    return condition()


# ---------------------------------------------------------------------------
# state machine (T-1)
# ---------------------------------------------------------------------------


def test_the_folder_cannot_change_while_a_run_is_active(window, monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr("gui.main_window.QFileDialog.getExistingDirectory", lambda *a, **k: str(other))
    before = window.folder
    window.state = STATE_MEASURING
    window._refresh_state()

    assert not window.folder_button.isEnabled()
    window._choose_folder()
    assert window.folder == before

    window.state = STATE_MEASURED
    window._refresh_state()
    window._choose_folder()
    assert window.folder == other


def test_correcting_does_not_depend_on_rejects(window):
    """A user report: with no rejects the correction looked unavailable."""
    window.threshold_slider.setValue(50)
    _measured(window, [95.0, 92.0, 88.0])

    assert window._count(50)["rejects"] == 0
    assert window.enhance_button.isEnabled() and "(3)" in window.enhance_button.text()
    assert not window.main_button.isEnabled() and window.main_button.text() == "Nothing to move"
    assert "Correcting is still possible" in window.summary_label.text()


def test_the_correction_section_opens_after_a_complete_measurement_only(window):
    window.results = [_result(window.folder, "a.jpg", 90.0)]
    window._on_measure_finished(window.results, True)
    assert not window.enhance_section.is_open

    window._on_measure_finished(window.results, False)
    assert window.enhance_section.is_open and window.state == STATE_MEASURED


# ---------------------------------------------------------------------------
# moving (T-5)
# ---------------------------------------------------------------------------


def test_survivors_stay_on_the_grid_after_moving(window, monkeypatch):
    window.threshold_slider.setValue(90)
    _measured(window, [95.0, 60.0, 92.0])
    _answer(monkeypatch)

    window._move_files()

    assert len(window.results) == 2 and len(window.cards) == 2
    assert window.state == STATE_MEASURED and window.enhance_button.isEnabled()
    assert (window.folder / config.FOLDER_REJECTS / "img1.jpg").exists()
    assert window.undo_button.isEnabled()


def test_duplicates_alone_can_be_moved(window, monkeypatch):
    """C-2: four marked duplicates and no rejects must still be movable."""
    window.threshold_slider.setValue(20)
    _measured(window, [80.0, 70.0, 75.0, 72.0, 90.0])
    for result in window.results[:4]:
        result.duplicate = True
    window._refresh_state()

    assert window.main_button.isEnabled()
    assert window.main_button.text() == "Move 4 duplicates"
    assert "Duplicates: 4" in window.summary_label.text()

    _answer(monkeypatch)
    window._move_files()

    assert len(list((window.folder / config.FOLDER_DUPLICATES).iterdir())) == 4
    assert len(window.results) == 1


def test_undo_asks_first(window, monkeypatch):
    """S-3: no undo without a confirmation naming what will happen."""
    _measured(window, [10.0])
    move_to_rejects(window.folder, window.results, 50)
    window._refresh_undo()
    window.state = STATE_MEASURED
    asked = []
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.question",
        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.No,
    )

    window._undo()

    assert asked and "move 1 file back" in asked[0]
    assert has_undoable_run(window.folder)


# ---------------------------------------------------------------------------
# correction settings and reports
# ---------------------------------------------------------------------------


def test_slider_defaults(window):
    assert window.wb_slider.value() == 0 and "(off)" in window.wb_label.text()
    assert window.brightness_slider.value() == 100 and "Brightness" in window.brightness_label.text()


def test_batch_and_preview_reports_agree():
    """EnhanceProgress (batch) and EnhanceResult (preview) are separate classes;
    they must reach the same conclusion on every relevant combination."""
    import numpy as np

    from core.enhance import EnhanceResult, enhance_image

    flat_dull = np.zeros((240, 320, 3), np.uint8)
    rng = np.random.default_rng(20260803)
    for i in range(8):
        flat_dull[:, i * 40:(i + 1) * 40] = rng.integers(0, 255, 3)
    flat_dull = np.clip(flat_dull.astype(np.float32) * 0.30 + 110.0, 0, 255).astype(np.uint8)
    corrected = enhance_image(flat_dull, 0.5, 30.0)
    assert corrected.denoise_h == 0.0 and corrected.sharpen_amount == 0.0
    assert corrected.tone_changed  # only colour/contrast acted

    base = dict(sigma=1.0, denoise_h=0.0, sharpen_amount=0.0, gains=(1.0, 1.0, 1.0), gamma=1.0)
    cases = [{}, {"gains": (1.0, 1.0, 0.9)}, {"gamma": 1.2}, {"clahe": 1.4}, {"saturation": 1.2},
             {"denoise_h": 3.0}, {"sharpen_amount": 0.5},
             {"clahe": corrected.clahe, "saturation": corrected.saturation}]
    for case in cases:
        fields = dict(base, **case)
        full, batch = EnhanceResult(image=None, **fields), EnhanceProgress(path=None, **fields)
        assert batch.tone_changed == full.tone_changed, case
        assert batch.anything_changed == full.anything_changed, case


def test_a_failing_save_is_reported_and_the_window_recovers(window, monkeypatch, caplog):
    """T-8: an exception in the worker becomes an error report plus a logged
    traceback; the window returns to the measured state."""
    def broken(folder, items):
        raise OSError("the undo log cannot be written")

    monkeypatch.setattr("gui.enhance_worker.save_corrected_batch", broken)
    worker = EnhanceWorker(window.folder, [])
    reports = []
    worker.finished_all.connect(lambda report, cancelled: reports.append(report))
    with caplog.at_level(logging.ERROR):
        worker.run()   # synchronously, in this thread

    assert "cannot be written" in reports[0].error
    assert "Traceback" in caplog.text

    shown = []
    monkeypatch.setattr("gui.main_window.QMessageBox.warning", lambda *a, **k: shown.append(a[2]))
    window.state = STATE_ENHANCING
    window._on_enhance_finished(reports[0], False)
    assert window.state == STATE_MEASURED and shown


def test_closing_during_a_run_does_not_block(window, monkeypatch, tmp_path):
    """C-15: the window disappears at once; the real close follows when the worker stops."""
    for i in range(3):
        Image.new("RGB", (64, 64), "gray").save(tmp_path / ("p%d.jpg" % i))
    real = worker_module.measure_item

    def slow(item):
        time.sleep(0.6)
        return real(item)

    monkeypatch.setattr(worker_module, "measure_item", slow)
    window.set_folder(tmp_path)
    window.show()
    window._start_measuring()

    started = time.monotonic()
    window.close()
    assert time.monotonic() - started < 0.5
    assert window.isHidden() and window._closing

    assert _wait_until(lambda: window.worker is None and not window.status_panel._timer.isActive())


# ---------------------------------------------------------------------------
# dialogs (T-7, T-12, T-16)
# ---------------------------------------------------------------------------


def test_preview_cancel_keeps_the_settings_and_dialogs_are_released(window, monkeypatch):
    _measured(window, [80.0])

    def cancel(self):
        assert callable(self.result)  # QDialog.result() is not shadowed
        self.sliders[0].setValue(300)
        return QDialog.Rejected

    monkeypatch.setattr(PreviewDialog, "exec", cancel)
    for _ in range(3):
        window._open_preview()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    assert window.denoise_slider.value() == 100
    assert window.findChildren(PreviewDialog) == []

    monkeypatch.setattr(PreviewDialog, "exec", lambda self: (self.sliders[0].setValue(200), QDialog.Accepted)[1])
    window._open_preview()
    assert window.denoise_slider.value() == 200


def test_dialogs_fit_the_smallest_and_the_default_window(window):
    _measured(window, [80.0])
    for width, height in ((960, 640), (1280, 820)):
        window.resize(width, height)
        for dialog in (PreviewDialog(window.results[0], 100, 100, 100, parent=window), AboutDialog(window)):
            dialog.show()
            QApplication.processEvents()
            card = dialog.card
            assert card.sizeHint().height() <= dialog.height() - 2 * dialog.MARGIN, (width, type(dialog))
            assert card.sizeHint().width() <= dialog.width() - 2 * dialog.MARGIN, (width, type(dialog))
            dialog.close()
            dialog.deleteLater()


def test_the_signature_reads_the_single_version():
    button = SignatureButton()
    assert button.text() == "psz // v%s" % __version__


# ---------------------------------------------------------------------------
# faces and duplicate marks
# ---------------------------------------------------------------------------


def _one_series(window, scores):
    _measured(window, scores)
    for i, result in enumerate(window.results):
        result.phash, result.captured = 0b1010, float(i)
    window._detect_series()


def test_after_face_analysis_the_duplicate_marks_follow_the_new_ranking(window):
    from core import faces as faces_module
    from core import series

    _one_series(window, [60.0, 70.0])
    a, b = window.results
    series.keep_best_n(window.series, 1)
    assert b.duplicate is False and a.duplicate is True

    # the faces reverse the ranking: a has a sharp face, b a soft one
    for result, face_score in ((a, 100.0), (b, 0.0)):
        face = faces_module.Face(box=(0, 0, 100, 100), right_eye=(30.0, 40.0), left_eye=(70.0, 40.0),
                                 nose=(50.0, 60.0), confidence=0.99)
        face.sharpness = face_score
        result.faces = faces_module.FaceReport(faces=[face])

    window._on_faces_finished(2, False)

    assert window.series[0].best is a and a.duplicate is False and b.duplicate is True


def test_face_analysis_never_marks_duplicates_unasked(window):
    _one_series(window, [60.0, 70.0])
    window._on_faces_finished(0, False)
    assert not any(r.duplicate for r in window.results)
