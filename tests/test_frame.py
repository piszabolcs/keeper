# Keeper - hit-test zones of the custom window frame. The caption is reported
# as HTCAPTION (the OS drags the window), never via startSystemMove() - that
# swallowed the next button click after every drag.

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from gui import frame  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = frame.FramelessWindow()
    w.setWindowTitle("Keeper")
    w.resize(1000, 700)
    bar = frame.TitleBar(w)
    w.set_caption_bar(bar)
    bar.setGeometry(0, 0, w.width(), frame.TITLE_BAR_HEIGHT)
    w.show()
    yield w
    w.close()


def _zone(window, x, y):
    """Hit test of a window-local point, given in logical global coordinates."""
    return window._hit_test(window.mapToGlobal(frame.point(x, y)))


def test_edges_and_corners_resize(window):
    w, h = window.width(), window.height()
    assert _zone(window, 1, 1) == frame.HTTOPLEFT
    assert _zone(window, w - 1, 1) == frame.HTTOPRIGHT
    assert _zone(window, 1, h - 1) == frame.HTBOTTOMLEFT
    assert _zone(window, w - 1, h - 1) == frame.HTBOTTOMRIGHT
    assert _zone(window, 1, h // 2) == frame.HTLEFT
    assert _zone(window, w - 1, h // 2) == frame.HTRIGHT
    assert _zone(window, w // 2, 1) == frame.HTTOP
    assert _zone(window, w // 2, h - 1) == frame.HTBOTTOM


def test_the_inside_belongs_to_the_program(window):
    assert _zone(window, window.width() // 2, window.height() // 2) == frame.HTCLIENT


def test_the_empty_caption_is_a_handle_but_its_buttons_are_not(window):
    y = frame.TITLE_BAR_HEIGHT // 2
    assert _zone(window, window.width() // 2, y) == frame.HTCAPTION

    bar = window._caption_bar
    for button in (bar.minimize_button, bar.maximize_button, bar.close_button):
        assert _zone(window, button.geometry().center().x(), y) == frame.HTCLIENT, button.toolTip()


def test_no_caption_bar_yet_does_not_crash(app):
    w = frame.FramelessWindow()
    w.resize(600, 400)
    w.show()
    try:
        assert _zone(w, 300, 200) == frame.HTCLIENT
    finally:
        w.close()


def test_the_caption_does_not_handle_mouse_events_itself():
    """As long as these are not overridden, Qt's mouse state cannot get stuck."""
    assert frame.TitleBar.mousePressEvent is QWidget.mousePressEvent
    assert frame.TitleBar.mouseDoubleClickEvent is QWidget.mouseDoubleClickEvent
