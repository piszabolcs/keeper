# Keeper - quick check of a single photo (T-22): verdict within a second,
# "Save corrected copy as", nothing moved or logged; folder/photo routing.

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import config  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.quick_check import QuickCheckDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _wait_until(condition, timeout=10.0):
    loop, timer = QEventLoop(), QTimer()
    deadline = time.monotonic() + timeout
    timer.timeout.connect(lambda: (condition() or time.monotonic() > deadline) and loop.quit())
    timer.start(10)
    loop.exec()
    timer.stop()
    return condition()


def _photo(path, size=1600):
    """A sharp checkerboard-like photo with shooting data."""
    image = np.zeros((size * 3 // 4, size, 3), np.uint8)
    square = size // 16
    for y in range(0, image.shape[0], square):
        for x in range(0, size, square):
            if ((x // square) + (y // square)) % 2 == 0:
                image[y:y + square, x:x + square] = (220, 200, 180)
    picture = Image.fromarray(image)
    exif = picture.getexif()
    exif.get_ifd(0x8769)[0x8827] = 400
    picture.save(path, quality=95, exif=exif)
    return path


def test_a_dropped_photo_gets_a_verdict_within_a_second_and_saves_as(app, tmp_path, monkeypatch):
    source = _photo(tmp_path / "single.jpg")
    original = source.read_bytes()

    started = time.monotonic()
    dialog = QuickCheckDialog(source, 50, (100, 100, 100, 0, 100))
    dialog.show()
    assert _wait_until(lambda: dialog.photo is not None, timeout=5.0)
    assert time.monotonic() - started < 1.0
    assert "threshold 50" in dialog.verdict_label.text()
    assert dialog.save_button.isEnabled()

    target = tmp_path / ("single%s.jpg" % config.QUICK_SAVE_SUFFIX)
    monkeypatch.setattr("gui.quick_check.QFileDialog.getSaveFileName", lambda *a, **k: (str(target), ""))
    dialog._save_as()
    assert _wait_until(lambda: dialog.status_label.text().startswith("Saved"), timeout=30.0)

    assert target.exists() and source.read_bytes() == original
    with Image.open(target) as saved:
        assert saved.getexif().get_ifd(0x8769).get(0x8827) == 400
    # quick mode never moves files or writes the undo log
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([source.name, target.name])
    dialog.close()


def test_the_original_can_never_be_the_save_target(app, tmp_path, monkeypatch):
    source = _photo(tmp_path / "single.jpg")
    dialog = QuickCheckDialog(source, 50, (100, 100, 100, 0, 100))
    assert _wait_until(lambda: dialog.photo is not None)
    monkeypatch.setattr("gui.quick_check.QFileDialog.getSaveFileName", lambda *a, **k: (str(source), ""))

    dialog._save_as()

    assert "never overwritten" in dialog.status_label.text()
    dialog.close()


def test_folders_and_photos_are_routed(app, tmp_path, monkeypatch):
    window = MainWindow()
    opened = []
    monkeypatch.setattr(window, "open_quick_check", lambda path: opened.append(path))
    photo = _photo(tmp_path / "a.jpg", size=320)
    (tmp_path / "notes.txt").write_text("x")

    window.open_path(tmp_path)
    window.open_path(photo)
    window.open_path(tmp_path / "notes.txt")

    assert window.folder == tmp_path
    assert opened == [photo]
    window.close()
