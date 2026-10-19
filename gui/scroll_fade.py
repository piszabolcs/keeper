"""A scroll area whose content fades out at the top and bottom edge.

Qt cannot mask a widget with a gradient, so two thin overlays sit on the
viewport's edges and paint the background colour fading into transparency.
Each one shows only while there is more content in its direction, so the
first and last lines are never dimmed when nothing is hidden. The overlays
ignore the mouse, clicks go straight through to the content.
"""
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget

FADE_HEIGHT = 26


class _EdgeFade(QWidget):
    def __init__(self, parent, color: QColor, top: bool):
        super().__init__(parent)
        self._color = color
        self._top = top
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.hide()

    def paintEvent(self, event):
        gradient = QLinearGradient(0, 0, 0, self.height())
        solid = QColor(self._color)
        clear = QColor(self._color)
        clear.setAlpha(0)
        gradient.setColorAt(0.0, solid if self._top else clear)
        gradient.setColorAt(1.0, clear if self._top else solid)
        painter = QPainter(self)
        painter.fillRect(self.rect(), gradient)
        painter.end()


class FadingScrollArea(QScrollArea):
    """QScrollArea with soft fades at the edges; `top_color`/`bottom_color` are
    the colours of whatever lies behind the scroll area."""

    def __init__(self, top_color: str, bottom_color: str | None = None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea > QWidget > QWidget { background: transparent; }")
        self.top_fade = _EdgeFade(self, QColor(top_color), top=True)
        self.bottom_fade = _EdgeFade(self, QColor(bottom_color or top_color), top=False)
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._update_fades)
        bar.rangeChanged.connect(self._update_fades)
        self.viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.viewport() and event.type() == QEvent.Resize:
            self._place_fades()
        return super().eventFilter(watched, event)

    def _place_fades(self):
        vp = self.viewport().geometry()
        height = min(FADE_HEIGHT, vp.height() // 3)
        self.top_fade.setGeometry(vp.x(), vp.y(), vp.width(), height)
        self.bottom_fade.setGeometry(vp.x(), vp.bottom() - height + 1, vp.width(), height)
        self._update_fades()

    def _update_fades(self, *_):
        bar = self.verticalScrollBar()
        self.top_fade.setVisible(bar.value() > bar.minimum())
        self.bottom_fade.setVisible(bar.value() < bar.maximum())
        for fade in (self.top_fade, self.bottom_fade):
            fade.raise_()
