# Keeper - background activity panel: what runs, on which file, for how
# long, and how hard the machine works. (No GPU row: everything here runs on
# the CPU, a GPU line stuck at 0 % would only mislead.)

import collections
import time

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui import theme

# psutil is optional: activity and elapsed time always work, the hardware
# rows are a bonus
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

PANEL_MARGIN = 20
# a QLabel adds a thin frame to its text width; the spare pixels keep the
# panel exactly as wide as the left column
TEXT_SPARE = 4
TEXT_WIDTH = theme.PANEL_WIDTH - 2 * PANEL_MARGIN - TEXT_SPARE

REFRESH_MS = 1000
# counting threads is expensive on Windows (4-10 ms): every third refresh only
THREAD_ROUNDS = 3

# Activities are named after the algorithm, not a "model": this is classic
# image processing.
ACTIVITY_IDLE = "Idle"
ACTIVITY_MEASURING = "Measuring sharpness"
ACTIVITY_ENHANCING = "Denoising and sharpening"
ACTIVITY_FACES = "Finding faces"


def elided(text, widget, width):
    """text shortened to width with an ellipsis in the middle."""
    return widget.fontMetrics().elidedText(text, Qt.ElideMiddle, width)


def _accent(alpha):
    color = QColor(theme.ACCENT_LIME)
    color.setAlpha(alpha)
    return color


class Sparkline(QWidget):
    """Tiny CPU history: shows whether the load is rising, steady or ending."""

    HEIGHT = 34
    SAMPLES = 60

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setMinimumWidth(40)
        self._values = collections.deque(maxlen=self.SAMPLES)

    def add_value(self, value):
        self._values.append(max(0.0, min(100.0, float(value))))
        self.update()

    def clear_values(self):
        self._values.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width, height = self.width(), self.height()

        # a baseline, so an empty chart does not look broken
        painter.setPen(QPen(_accent(30), 1))
        painter.drawLine(0, height - 1, width, height - 1)

        points = self._points(width, height)
        if len(points) >= 2:
            fill = QPainterPath()
            fill.moveTo(points[0].x(), height)
            for p in points:
                fill.lineTo(p)
            fill.lineTo(points[-1].x(), height)
            fill.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.fillPath(fill, _accent(45))
            painter.setPen(QPen(QColor(theme.ACCENT_LIME), 1.6))
            painter.drawPolyline(QPolygonF(points))
        painter.end()

    def _points(self, width, height):
        """Fixed step, newest sample on the right edge: the line flows leftwards."""
        count = len(self._values)
        step = width / float(self.SAMPLES - 1)
        usable = max(1.0, height - 2.0)
        return [
            QPointF(width - (count - 1 - i) * step, (height - 1.0) - (v / 100.0) * usable)
            for i, v in enumerate(self._values)
        ]


class StatusPanel(QFrame):
    """Glass panel under the controls with the background information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glassPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)

        # full texts are kept: labels only ever get the elided version
        self._activity = ACTIVITY_IDLE
        self._detail = ""
        self._started = None   # None = idle, the clock does not run
        self._process = None
        self._cores = 0
        self._thread_countdown = 0
        self._start_psutil()

        self._build()

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self.set_idle()

    def _start_psutil(self):
        """The process handle is fetched ONCE (re-opening it every second is waste)."""
        if not PSUTIL_AVAILABLE:
            return
        try:
            self._process = psutil.Process()
            self._cores = psutil.cpu_count(logical=True) or 0
            psutil.cpu_percent(interval=None)  # warm-up: the first call returns 0.0
        except Exception:
            self._process = None

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN)
        layout.setSpacing(10)

        title = QLabel("Background activity")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.activity_label = QLabel()
        self.activity_label.setWordWrap(False)
        self.activity_label.setStyleSheet("font-size: 15px; font-weight: 600; color: %s;" % theme.TEXT_PRIMARY)
        layout.addWidget(self.activity_label)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("muted")
        self.detail_label.setWordWrap(False)
        layout.addWidget(self.detail_label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 13px; font-weight: 600; color: %s;" % theme.ACCENT_LEMON)
        row.addWidget(self.progress_label)
        row.addStretch()
        self.elapsed_label = QLabel()
        self.elapsed_label.setObjectName("muted")
        self.elapsed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.elapsed_label)
        layout.addLayout(row)

        layout.addSpacing(4)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: %s; border: none;" % theme.GLASS_BORDER)
        layout.addWidget(line)
        layout.addSpacing(2)
        layout.addWidget(self._build_machine())

    def _build_machine(self):
        """Hardware rows in one box, so they vanish together without psutil."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # whole-machine CPU: the user wants to know if there is capacity left
        row, self.cpu_label = self._data_row("Processor", theme.ACCENT_LIME)
        layout.addLayout(row)
        self.sparkline = Sparkline()
        layout.addWidget(self.sparkline)
        layout.addSpacing(2)
        # our own memory first: corrections hold large images
        row, self.memory_label = self._data_row("Memory")
        layout.addLayout(row)
        row, self.threads_label = self._data_row("Threads")
        layout.addLayout(row)

        self.no_data_label = QLabel("Machine data is not available.")
        self.no_data_label.setObjectName("muted")
        self.no_data_label.setWordWrap(True)
        self.no_data_label.setVisible(False)
        layout.addWidget(self.no_data_label)

        if self._process is None:
            for label in (self.cpu_label, self.memory_label, self.threads_label):
                label.setText("–")
            self.sparkline.setVisible(False)
            self.no_data_label.setVisible(True)
        return box

    def _data_row(self, name, value_color=None):
        """A "name ..... value" row with the value right-aligned."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name_label = QLabel(name)
        name_label.setObjectName("muted")
        row.addWidget(name_label)
        row.addStretch()
        value = QLabel("–")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if value_color is None:
            value.setObjectName("secondary")
        else:
            value.setStyleSheet("font-weight: 600; color: %s;" % value_color)
        row.addWidget(value)
        return row, value

    # -----------------------------------------------------------------
    # public interface
    # -----------------------------------------------------------------

    def set_activity(self, activity, detail=None):
        """What runs now (detail: typically the current file name). The clock
        restarts only when the activity itself changes."""
        if activity != self._activity or self._started is None:
            self._started = time.monotonic()
            self._thread_countdown = 0  # thread count jumps between phases
        self._activity = activity
        self._detail = detail or ""
        self._show_texts()
        self._show_elapsed()

    def set_idle(self, message=ACTIVITY_IDLE):
        self._activity = message
        self._detail = ""
        self._started = None
        self._show_texts()
        self._show_elapsed()
        self.set_progress(None, None)

    def set_progress(self, done, total):
        """The "12 / 63" pair; hidden for None."""
        if done is None or total is None:
            self.progress_label.setText("")
            self.progress_label.setVisible(False)
            return
        self.progress_label.setText("%d / %d" % (done, total))
        self.progress_label.setVisible(True)

    def start_monitoring(self):
        self._refresh()
        self._timer.start()

    def stop_monitoring(self):
        self._timer.stop()

    # -----------------------------------------------------------------
    # refreshing
    # -----------------------------------------------------------------

    def _refresh(self):
        self._show_elapsed()
        self._show_machine()

    def _show_elapsed(self):
        if self._started is None:
            self.elapsed_label.setText("")
            self.elapsed_label.setVisible(False)
            return
        seconds = int(time.monotonic() - self._started)
        self.elapsed_label.setText("%d:%02d" % (seconds // 60, seconds % 60))
        self.elapsed_label.setVisible(True)

    def _show_machine(self):
        if self._process is None:
            return
        try:
            # interval=None returns immediately (an interval would block the GUI thread)
            cpu = psutil.cpu_percent(interval=None)
            own_mb = self._process.memory_info().rss / (1024.0 * 1024.0)
            system_percent = psutil.virtual_memory().percent
        except Exception:
            return  # a status panel must never stop the processing

        self.cpu_label.setText("%d%%" % round(cpu))
        self.sparkline.add_value(cpu)
        self.memory_label.setText("%d MB · machine %d%%" % (round(own_mb), round(system_percent)))

        self._thread_countdown -= 1
        if self._thread_countdown > 0:
            return
        self._thread_countdown = THREAD_ROUNDS
        try:
            threads = self._process.num_threads()
        except Exception:
            return
        self.threads_label.setText("%d (%d cores)" % (threads, self._cores) if self._cores else "%d" % threads)

    def _show_texts(self):
        """Labels always get the elided text; the raw text would widen the panel."""
        width = self._text_width()
        self.activity_label.setText(elided(self._activity, self.activity_label, width))
        self.activity_label.setToolTip(self._activity)
        self.detail_label.setText(elided(self._detail, self.detail_label, width))
        self.detail_label.setToolTip(self._detail)
        # empty labels still take a line; hide them while there is nothing to show
        self.detail_label.setVisible(bool(self._detail))
        self.progress_label.setVisible(bool(self.progress_label.text()))
        self.elapsed_label.setVisible(bool(self.elapsed_label.text()))

    def _text_width(self):
        actual = self.width() - 2 * PANEL_MARGIN - TEXT_SPARE
        return TEXT_WIDTH if actual < 40 else min(TEXT_WIDTH, actual)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_texts()
