# Keeper - correction preview on one photo.
#
# Shows a before/after pair of the most detailed crop at 1:1 while the
# sliders act live. The crop is corrected with the WHOLE frame's statistics
# (core/preview.py), so it shows what the saved copy will get. Loading and
# computing run in background threads (a 480 px crop takes ~0.5 s to denoise).

import logging

import cv2
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import config
from core import preview
from core.enhance import load_bgr_full
from core.noise import read_iso
from gui import theme

log = logging.getLogger(__name__)

# shared by the main window and the preview
WB_TOOLTIP = (
    "OFF BY DEFAULT, for a good reason.\n\n"
    "A camera JPEG already carries the camera's own white balance. An\n"
    "after-the-fact estimate cannot tell warm LIGHT from a warm SUBJECT: on the\n"
    "reference set it neutralised brick to grey and faded a blue sky to cyan.\n\n"
    "Against a strong indoor colour cast it does help - raise it and the\n"
    "preview shows the effect immediately."
)

# unparented threads that are still running; a closed dialog does not wait
# for them (and never blocks the UI), they finish and are released here
_running = set()


def start_detached(thread):
    _running.add(thread)
    thread.finished.connect(lambda: _running.discard(thread))
    thread.start()


def bgr_to_pixmap(bgr):
    """A QPixmap copy of an OpenCV BGR image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())  # copy: rgb is freed on return


class _Loader(QThread):
    """Full-resolution load + crop + full-frame statistics."""

    done = Signal(object, object, str)  # PreviewInput, iso, error

    def __init__(self, path, size):
        super().__init__()
        self.path, self.size = path, size

    def run(self):
        try:
            source = preview.prepare(load_bgr_full(self.path), self.size)
            iso = read_iso(self.path)
        except Exception as exc:
            log.warning("preview of %s could not be loaded: %s", self.path, exc)
            self.done.emit(None, None, str(exc))
            return
        self.done.emit(source, iso, "")


class _Renderer(QThread):
    """One correction of the preview crop."""

    done = Signal(object, str)  # EnhanceResult, error

    def __init__(self, source, score, iso, values, learned):
        super().__init__()
        self.source, self.score, self.iso, self.values, self.learned = source, score, iso, values, learned

    def run(self):
        denoise, sharpen, brightness, wb, color = self.values
        try:
            result = preview.render(
                self.source, self.score, denoise, sharpen, brightness, self.iso, wb, self.learned, color
            )
        except Exception as exc:
            log.exception("preview correction failed")
            self.done.emit(None, str(exc))
            return
        self.done.emit(result, "")


class PreviewDialog(QDialog):
    """
    Frameless overlay with a centred card. "Use these settings" (accept)
    hands the slider values back to the caller; Cancel, Esc and a click
    outside the card leave the caller's settings unchanged.
    """

    MARGIN = 24
    CONTROLS_WIDTH = 260
    MIN_IMAGE = 160
    # width of everything in the card except the two images
    FIXED_WIDTH = 2 * 26 + 18 + 24 + CONTROLS_WIDTH
    # height of title row, captions and margins around the images
    FIXED_HEIGHT = 22 + 20 + 50 + 28

    def __init__(self, photo, denoise_percent, sharpen_percent, brightness_percent,
                 wb_percent=0, learned=False, color_percent=100, parent=None):
        super().__init__(parent)
        self.photo = photo   # not "result": that would shadow QDialog.result()
        self.learned = learned
        self.source = None
        self.iso = None
        self._renderer = None
        self._again = False

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        if parent is not None:
            self.setGeometry(parent.geometry())

        # the crop adapts to the window, never the other way round: 1:1 is the
        # point of the preview, so a small window gets a smaller crop
        self.image_size = self._image_size(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        self.card = self._build_card(
            (denoise_percent, sharpen_percent, brightness_percent, wb_percent, color_percent)
        )
        outer.addWidget(self.card, 0, Qt.AlignCenter)

        # sliders do not recompute on every tick of a drag
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(config.PREVIEW_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._render)

        loader = _Loader(photo.path, self.image_size)
        loader.done.connect(self._loaded)
        start_detached(loader)

    def _image_size(self, parent):
        if parent is None:
            return config.PREVIEW_CROP
        width = (parent.width() - 2 * self.MARGIN - self.FIXED_WIDTH) // 2
        height = parent.height() - 2 * self.MARGIN - self.FIXED_HEIGHT
        return max(self.MIN_IMAGE, min(config.PREVIEW_CROP, width, height))

    # -----------------------------------------------------------------
    # building
    # -----------------------------------------------------------------

    def _build_card(self, values):
        card = QWidget()
        card.setObjectName("dialogCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(26, 22, 26, 20)
        layout.setSpacing(12)

        title = QLabel("Correction preview")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.file_label = QLabel(self.photo.path.name)
        self.file_label.setObjectName("muted")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        body = QHBoxLayout()
        body.setSpacing(24)
        images = QHBoxLayout()
        images.setSpacing(18)
        self.before_label = self._image_box(images, "Original")
        self.after_label = self._image_box(images, "Corrected")
        body.addLayout(images)
        body.addWidget(self._build_controls(values), 0, Qt.AlignTop)
        layout.addLayout(body)
        return card

    def _image_box(self, row, caption):
        column = QVBoxLayout()
        column.setSpacing(6)
        title = QLabel(caption)
        title.setObjectName("muted")
        title.setAlignment(Qt.AlignCenter)
        column.addWidget(title)
        label = QLabel("Loading…")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setFixedSize(self.image_size, self.image_size)
        label.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 8px;")
        column.addWidget(label)
        column.addStretch()
        row.addLayout(column)
        return label

    def _build_controls(self, values):
        controls = QWidget()
        controls.setFixedWidth(self.CONTROLS_WIDTH)
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(4)
        self.sliders, self.slider_labels = [], []
        for row, value in enumerate(values):
            label = QLabel()
            label.setObjectName("muted")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, config.ENHANCE_SCALE_MAX)
            slider.setValue(value)
            slider.setSingleStep(25)
            slider.setPageStep(25)
            slider.valueChanged.connect(self._slider_moved)
            grid.addWidget(label, row * 2, 0)
            grid.addWidget(slider, row * 2 + 1, 0)
            self.sliders.append(slider)
            self.slider_labels.append(label)
        self.sliders[3].setToolTip(WB_TOOLTIP)
        layout.addLayout(grid)
        layout.addSpacing(8)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("secondary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setMinimumHeight(self.summary_label.fontMetrics().lineSpacing() * 7)
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.summary_label)
        layout.addSpacing(8)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("secondary")
        cancel.setCursor(Qt.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.accept_button = QPushButton("Use these settings")
        self.accept_button.setObjectName("primary")
        self.accept_button.setCursor(Qt.PointingHandCursor)
        self.accept_button.setDefault(True)
        self.accept_button.clicked.connect(self.accept)
        buttons.addWidget(self.accept_button)
        layout.addLayout(buttons)

        self._update_labels()
        return controls

    # -----------------------------------------------------------------
    # events
    # -----------------------------------------------------------------

    def _update_labels(self):
        names = ("Denoising", "Sharpening", "Brightness", "White balance", "Colour & local contrast")
        for name, slider, label in zip(names, self.sliders, self.slider_labels):
            off = " (off)" if slider.value() == 0 and name in ("White balance", "Colour & local contrast") else ""
            label.setText("%s: %d%%%s" % (name, slider.value(), off))

    def _slider_moved(self, _value=None):
        self._update_labels()
        if self.source is not None:
            self._debounce.start()

    def _loaded(self, source, iso, error):
        if error:
            self.before_label.setText("Could not be loaded:\n%s" % error)
            self.after_label.setText("")
            return
        self.source, self.iso = source, iso
        crop = source.crop
        self.before_label.setPixmap(bgr_to_pixmap(crop))
        self.file_label.setText(
            "%s · the most detailed %d×%d pixels at 1:1 (from %d, %d)"
            % (self.photo.path.name, crop.shape[1], crop.shape[0], source.location[0], source.location[1])
        )
        self._render()

    def _render(self):
        if self.source is None:
            return
        # one computation at a time; a newer slider value triggers one more round
        if self._renderer is not None and self._renderer.isRunning():
            self._again = True
            return
        self.summary_label.setText("Computing…")
        self._renderer = _Renderer(self.source, self.photo.score, self.iso, self.values(), self.learned)
        self._renderer.done.connect(self._rendered)
        start_detached(self._renderer)

    def _rendered(self, result, error):
        if error:
            self.summary_label.setText("The preview failed:\n%s" % error)
        else:
            self.after_label.setPixmap(bgr_to_pixmap(result.image))
            self.summary_label.setText(self._summary(result))
        if self._again:
            self._again = False
            self._render()

    def _summary(self, result):
        """
        What happened - and WHY a step did nothing. A bare zero looks like a
        broken program; "the frame is already clean" is an answer.
        """
        score = round(self.photo.score or 0)
        wb_off, color_off = self.sliders[3].value() == 0, self.sliders[4].value() == 0
        lines = ["Measured noise %.1f · score %d" % (result.sigma or 0.0, score)]

        lines.append("Denoising %.1f" % result.denoise_h if result.denoise_h > 0.0
                     else "No denoising (the frame is below the noise floor)")
        if result.sharpen_amount > 0.0:
            lines.append("Sharpening %.2f" % result.sharpen_amount)
        elif score < config.SHARPEN_SCORE_FLOOR:
            lines.append("No sharpening (below %d points the detail is gone)" % config.SHARPEN_SCORE_FLOOR)
        else:
            lines.append("No sharpening (slider at zero)")

        if abs(result.gamma - 1.0) > 1e-6:
            lines.append("Brightness: %s, gamma %.2f" % ("darken" if result.gamma > 1.0 else "brighten", result.gamma))
        else:
            lines.append("Brightness unchanged (already in the good band)")

        if any(abs(g - 1.0) > 1e-6 for g in result.gains):
            lines.append("White balance R/G/B %.2f/%.2f/%.2f" % (result.gains[2], result.gains[1], result.gains[0]))
        else:
            lines.append("White balance off" if wb_off else "White balance unchanged (neutral colours)")

        if result.clahe > 0.0:
            lines.append("Local contrast %.1f (flat frame)" % result.clahe)
        else:
            lines.append("Local contrast off" if color_off else "No local contrast (contrasty enough)")

        if result.saturation > 1.0 + 1e-6:
            lines.append("Saturation %+.0f%%" % ((result.saturation - 1.0) * 100))
        else:
            lines.append("Saturation off" if color_off else "Saturation unchanged (saturated enough)")

        return "\n".join(lines)

    def values(self):
        """(denoise, sharpen, brightness, white balance, colour) percentages."""
        return tuple(slider.value() for slider in self.sliders)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 115))
        painter.end()

    def mousePressEvent(self, event):
        # outside the card = dismiss (same as Esc: the settings stay unchanged)
        if not self.card.geometry().contains(event.position().toPoint()):
            self.reject()
            return
        super().mousePressEvent(event)
