# Keeper - quick check of a single photo.
#
# Drop one photo (or open it) and get an instant verdict: score, part scores,
# what the shooting data says and the faces - without choosing a folder.
# "Correct…" opens the usual preview; "Save corrected copy as…" writes a
# corrected JPEG (EXIF and ICC kept) wherever the user wants. Nothing is
# moved and nothing is logged in this mode.

import logging
import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import config
from core import exif, faces
from core.enhance import Cancelled, enhance_image, load_bgr_full
from core.noise import read_iso, sigma_from_gray
from core.sharpness import is_kept, measure_sharpness
from gui import theme
from gui.preview_dialog import PreviewDialog, start_detached
from gui.result_card import thumbnail_image
from utils.save_ops import quick_save_name, save_corrected_as

log = logging.getLogger(__name__)


class _Measure(QThread):
    """Measures the photo, loads a large preview and analyses faces."""

    done = Signal(object, object)  # SharpnessResult, QImage or None

    def __init__(self, path, image_size):
        super().__init__()
        self.path, self.image_size = path, image_size

    def run(self):
        result = measure_sharpness(self.path)  # never raises
        image = None
        try:
            image = thumbnail_image(result.path, self.image_size)
        except Exception:
            pass
        if result.score is not None and faces.model_available():
            result.faces = faces.analyse_path(result.path)
        self.done.emit(result, image)


class _Save(QThread):
    """Full-resolution correction of the photo, saved to the chosen file."""

    done = Signal(str, str)  # target, error

    def __init__(self, result, values, learned, target, cancel):
        super().__init__()
        self.result, self.values, self.learned, self.target, self.cancel = result, values, learned, target, cancel

    def run(self):
        denoise, sharpen, brightness, wb, color = self.values
        path = self.result.path
        try:
            bgr = load_bgr_full(path)
            corrected = enhance_image(
                bgr, sigma_from_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)), self.result.score,
                denoise, sharpen, brightness, read_iso(path), wb, self.learned, color, cancel=self.cancel,
            )
            save_corrected_as(path, corrected.image, self.target)
        except Cancelled:
            self.done.emit("", "cancelled")
            return
        except Exception as exc:
            log.exception("saving a corrected copy of %s failed", path)
            self.done.emit("", str(exc))
            return
        log.info("quick check: saved %s", self.target)
        self.done.emit(str(self.target), "")


class QuickCheckDialog(QDialog):
    """Frameless overlay with the verdict card; X, Esc or a click outside closes it."""

    MARGIN = 24
    INFO_WIDTH = 300

    def __init__(self, path, threshold, values, learned=False, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.threshold = threshold
        self.values = tuple(values)
        self.learned = learned
        self.photo = None   # the SharpnessResult once measured
        self._cancel = threading.Event()

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        if parent is not None:
            self.setGeometry(parent.geometry())
        self.image_size = self._image_size(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        self.card = self._build_card()
        outer.addWidget(self.card, 0, Qt.AlignCenter)

        measure = _Measure(self.path, self.image_size)
        measure.done.connect(self._measured)
        start_detached(measure)

    def _image_size(self, parent):
        if parent is None:
            return 560
        room = parent.width() - 2 * self.MARGIN - 52 - 24 - self.INFO_WIDTH
        return max(240, min(560, room, parent.height() - 2 * self.MARGIN - 120))

    def _build_card(self):
        card = QWidget()
        card.setObjectName("dialogCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(26, 20, 20, 22)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Quick check")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        name = QLabel(self.path.name)
        name.setObjectName("muted")
        header.addWidget(name, 1)
        close = QPushButton("✕")
        close.setCursor(Qt.PointingHandCursor)
        close.setToolTip("Close (Esc)")
        close.setAccessibleName("Close")
        close.setFixedSize(28, 28)
        close.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: %s; font-size: 14px; }"
            "QPushButton:hover { color: %s; }" % (theme.TEXT_SECONDARY, theme.ACCENT_LEMON)
        )
        close.clicked.connect(self.reject)
        header.addWidget(close, 0, Qt.AlignTop)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(24)
        self.image_label = QLabel("Measuring…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(self.image_size, self.image_size * 3 // 4)
        self.image_label.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 8px;")
        body.addWidget(self.image_label, 0, Qt.AlignTop)
        body.addWidget(self._build_info(), 0, Qt.AlignTop)
        layout.addLayout(body)
        return card

    def _build_info(self):
        info = QWidget()
        info.setFixedWidth(self.INFO_WIDTH)
        layout = QVBoxLayout(info)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.score_label = QLabel()
        layout.addWidget(self.score_label)
        self.verdict_label = QLabel("Measuring…")
        self.verdict_label.setWordWrap(True)
        layout.addWidget(self.verdict_label)
        self.details_label = QLabel()
        self.details_label.setObjectName("secondary")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        layout.addSpacing(8)

        self.correct_button = QPushButton("Correct…")
        self.correct_button.setObjectName("secondary")
        self.correct_button.setCursor(Qt.PointingHandCursor)
        self.correct_button.setToolTip("Preview the correction on the most detailed crop")
        self.correct_button.clicked.connect(self._open_preview)
        layout.addWidget(self.correct_button)
        self.save_button = QPushButton("Save corrected copy as…")
        self.save_button.setObjectName("primary")
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.clicked.connect(self._save_as)
        layout.addWidget(self.save_button)
        self.status_label = QLabel()
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        for button in (self.correct_button, self.save_button):
            button.setEnabled(False)
        return info

    # -----------------------------------------------------------------
    # events
    # -----------------------------------------------------------------

    def _measured(self, result, image):
        self.photo = result
        if image is not None:
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText("No preview")

        if result.score is None:
            self.score_label.setText('<span style="font-size:30px; font-weight:600; color:%s;">–</span>' % theme.TEXT_MUTED)
            self.verdict_label.setText("This file cannot be read.")
            self.details_label.setText(result.error or "")
            return

        kept = is_kept(result, self.threshold)
        color = theme.ACCENT_LIME if kept else theme.TEXT_SECONDARY
        self.score_label.setText(
            '<span style="font-size:30px; font-weight:600; color:%s;">%d</span>'
            '<span style="font-size:12px; color:%s;">  / 100</span>' % (color, round(result.score), theme.TEXT_SECONDARY)
        )
        self.verdict_label.setText(
            '<span style="color:%s; font-weight:600;">%s</span>'
            % (color, ("Keep – sharp enough at threshold %d" if kept else "Reject – below threshold %d") % self.threshold)
        )
        self.details_label.setText(self._details(result))
        for button in (self.correct_button, self.save_button):
            button.setEnabled(True)

    def _details(self, result):
        lines = ["L %d · T %d · B %d" % (round(result.laplacian), round(result.tenengrad), round(result.brenner))]
        if abs(result.expectation or 0.0) >= 0.5:
            lines.append("Measured %d before the %s adjustment for the shooting conditions" % (
                round(result.raw_score), "strict" if result.expectation > 0 else "lenient"))
        shot = exif.describe(result.shot)
        if shot:
            lines.append(shot)
        lines.extend(result.warnings or [])
        report = result.faces
        if report is not None and report.faces:
            main = report.main
            text = "%d face%s" % (report.count, "" if report.count == 1 else "s")
            if main.sharpness is not None:
                text += ", main face %d" % round(main.sharpness)
                if main.sharpness < config.FACE_SOFT_THRESHOLD:
                    text += " – soft"
            lines.append(text)
        if result.low_resolution:
            lines.append("Low resolution (below %d px)" % config.WORK_RESOLUTION)
        return "\n".join(lines)

    def _open_preview(self):
        denoise, sharpen, brightness, wb, color = self.values
        dialog = PreviewDialog(self.photo, denoise, sharpen, brightness, wb, self.learned, color, self.parent() or self)
        try:
            if dialog.exec() == QDialog.Accepted:
                self.values = dialog.values()
                self.status_label.setText("The settings from the preview will be used.")
        finally:
            dialog.deleteLater()

    def _save_as(self):
        default = str(quick_save_name(self.photo.path))
        target, _ = QFileDialog.getSaveFileName(self, "Save corrected copy as", default, "JPEG (*.jpg)")
        if not target:
            return
        if Path(target).resolve() == self.photo.path.resolve():
            self.status_label.setText("Choose another name: the original is never overwritten.")
            return
        for button in (self.correct_button, self.save_button):
            button.setEnabled(False)
        self.status_label.setText("Saving… (a few seconds)")
        saver = _Save(self.photo, self.values, self.learned, Path(target), self._cancel)
        saver.done.connect(self._saved)
        start_detached(saver)

    def _saved(self, target, error):
        for button in (self.correct_button, self.save_button):
            button.setEnabled(True)
        if error:
            self.status_label.setText("Not saved: %s" % error)
        else:
            self.status_label.setText("Saved: %s" % Path(target).name)

    def done(self, code):
        self._cancel.set()   # a save still running is abandoned
        super().done(code)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 115))
        painter.end()

    def mousePressEvent(self, event):
        if not self.card.geometry().contains(event.position().toPoint()):
            self.reject()
            return
        super().mousePressEvent(event)
