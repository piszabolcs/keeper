# Keeper - one shot's card in the result grid: thumbnail, name, big score,
# the three part scores (to see which metric pulls the result), series, face
# and header hints, and the manual decision.

import io

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QLabel, QVBoxLayout

import config
from core import exif, raw_loader
from core.sharpness import is_kept
from gui import theme

THUMBNAIL_SIZE = 192
CARD_WIDTH = 208
TEXT_WIDTH = CARD_WIDTH - 16   # minus the inner margins

STATE_ERROR = "unreadable"
STATE_REJECT = "reject"
STATE_DUPLICATE = "duplicate"
STATE_KEPT = "kept"


def elided(text, widget, width):
    return widget.fontMetrics().elidedText(text, Qt.ElideMiddle, width)


def _to_qimage(img, size):
    """A PIL image as an upright QImage thumbnail (safe in worker threads)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    data = img.tobytes("raw", "RGB")
    # copy(): the QImage must own its bytes, data dies with this function
    return QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888).copy()


def thumbnail_image(path, size=THUMBNAIL_SIZE):
    """
    Thumbnail as a QImage. JPEGs decode at reduced size (draft); RAW files use
    the embedded preview and only fall back to a slow demosaic without one.
    Called by the measuring worker, so the GUI thread never decodes images.
    """
    if raw_loader.is_raw(path):
        data = raw_loader.preview_jpeg_bytes(path)
        if data is None:
            return _to_qimage(Image.fromarray(raw_loader.demosaic_rgb(path)), size)
        source = io.BytesIO(data)
    else:
        source = path

    with Image.open(source) as img:
        img.draft("RGB", (size, size))
        return _to_qimage(img, size)


class ResultCard(QFrame):
    # the user flipped the decision; the window refreshes its counters
    override_changed = Signal()
    # right click: open the correction preview for this shot
    preview_requested = Signal(object)

    # left click cycles: automatic -> always keep -> always reject -> automatic
    OVERRIDE_CYCLE = [None, True, False]

    def __init__(self, result, threshold, parent=None):
        super().__init__(parent)
        self.result = result
        self.threshold = threshold
        self.series_info = None
        self._shown = None   # last (state, manual) drawn - skip identical redraws

        self.setObjectName("resultCard")
        self.setFixedWidth(CARD_WIDTH)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 10)
        layout.setSpacing(6)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(THUMBNAIL_SIZE)
        self._show_thumbnail()
        # created once: every setGraphicsEffect call would delete the previous one
        self._opacity = QGraphicsOpacityEffect(self.image_label)
        self.image_label.setGraphicsEffect(self._opacity)
        layout.addWidget(self.image_label)

        # long names are elided: an unwrapped label would widen the fixed card
        self.name_label = QLabel(elided(result.path.name, self, TEXT_WIDTH))
        self.name_label.setObjectName("secondary")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setToolTip(result.path.name)
        layout.addWidget(self.name_label)

        # RAW / pair marker: a pair moves TWO files
        self.files_label = self._small_label(layout, "muted", self._files_text())
        self.series_label = self._small_label(layout)
        self.face_label = self._small_label(layout)
        # header-only hints (never reject): they explain WHY a frame may be soft
        hints = getattr(result, "warnings", None) or []
        self.hint_label = self._small_label(
            layout, None,
            '<span style="color:%s; font-size:10px;">%s</span>' % (theme.ACCENT_LEMON, " · ".join(hints))
            if hints else "",
        )

        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)

        self.parts_label = QLabel()
        self.parts_label.setObjectName("muted")
        self.parts_label.setAlignment(Qt.AlignCenter)
        self.parts_label.setWordWrap(True)
        self.parts_label.setToolTip(
            "L / T / B: Laplacian, Tenengrad and Brenner part scores (0-100).\n"
            "\"measured\": the score before the adjustment for the shooting conditions."
        )
        layout.addWidget(self.parts_label)

        tooltip = (
            "Left click: override the decision\n"
            "automatic → always keep → always reject\n\n"
            "Right click: correction preview of this shot"
        )
        details = exif.describe(getattr(result, "shot", None))
        if result.score is None:
            tooltip = "This file could not be read:\n%s" % (result.error or "unknown error")
        elif details:
            tooltip = "%s\n%s\n\n%s" % (result.path.name, details, tooltip)
        self.setToolTip(tooltip)

        self.set_threshold(threshold)

    def _small_label(self, layout, name=None, text=""):
        label = QLabel(text)
        if name:
            label.setObjectName(name)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)  # wraps, so no file name combination widens the card
        label.setVisible(bool(text))
        layout.addWidget(label)
        return label

    def _files_text(self):
        files = self.result.files or [self.result.path]
        if len(files) > 1:
            kinds = " + ".join(sorted({f.suffix.lstrip(".").upper() for f in files}))
            return "%s · %d files move together" % (kinds, len(files))
        return "RAW" if raw_loader.is_raw(files[0]) else ""

    def _show_thumbnail(self):
        image = getattr(self.result, "thumbnail", None)
        try:
            if image is None:
                image = thumbnail_image(self.result.path)
            self.image_label.setPixmap(QPixmap.fromImage(image))
        except Exception:
            self.image_label.setText("no preview")
            self.image_label.setObjectName("muted")

    def mousePressEvent(self, event):
        # preview on the RIGHT button so it never clashes with the override
        if event.button() == Qt.RightButton and self.result.score is not None:
            self.preview_requested.emit(self.result)
            return
        if event.button() == Qt.LeftButton and self.result.score is not None:
            position = self.OVERRIDE_CYCLE.index(self.result.override)
            self.result.override = self.OVERRIDE_CYCLE[(position + 1) % len(self.OVERRIDE_CYCLE)]
            self.set_threshold(self.threshold)
            self.override_changed.emit()
            return
        super().mousePressEvent(event)

    def set_series(self, info):
        """Series position; the best is highlighted in the KEEP colour."""
        self.series_info = info
        if info is None:
            self.series_label.clear()
            self.series_label.setVisible(False)
            return
        text = "Series %d · %d/%d" % (info.index, info.position, info.size)
        color = theme.TEXT_MUTED
        if info.is_best:
            text += " ★ best"
            color = theme.ACCENT_LIME
        self.series_label.setText('<span style="color:%s;">%s</span>' % (color, text))
        self.series_label.setVisible(True)

    def set_faces(self, report):
        """Face result; a soft main face is a highlighted hint, never a verdict."""
        if report is None or not report.faces:
            self.face_label.clear()
            self.face_label.setVisible(False)
            return
        main = report.main
        if main.sharpness is not None and main.sharpness < config.FACE_SOFT_THRESHOLD:
            text, color = "⚠ soft face (%d points)" % round(main.sharpness), theme.ACCENT_LEMON
        else:
            text, color = "%d face%s" % (report.count, "" if report.count == 1 else "s"), theme.TEXT_MUTED
            if main.sharpness is not None:
                text += " · face sharpness %d" % round(main.sharpness)
        self.face_label.setText('<span style="color:%s;">%s</span>' % (color, text))
        self.face_label.setVisible(True)

    def state(self, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        if self.result.score is None:
            return STATE_ERROR
        if not is_kept(self.result, threshold):
            return STATE_REJECT
        if self.result.duplicate and self.result.override is not True:
            return STATE_DUPLICATE
        return STATE_KEPT

    def set_threshold(self, threshold):
        """
        Recolours for a new threshold - no re-measuring, the scores exist.
        Cards whose state did not change are skipped entirely, so a slider
        step costs only the cards that actually flip (C-5).
        """
        self.threshold = threshold
        state = self.state(threshold)
        manual = self.result.override is not None
        if self._shown == (state, manual, self.result.duplicate):
            return
        self._shown = (state, manual, self.result.duplicate)

        if state == STATE_ERROR:
            self._paint(state, theme.TEXT_MUTED, 0.45)
            self.parts_label.setText("Cannot read this file")
            return

        self.parts_label.setText(self._parts_text())
        if state == STATE_REJECT:
            # rejects are not red: rejecting is a decision, not an error
            self._paint(state, theme.TEXT_SECONDARY, 0.45)
        elif state == STATE_DUPLICATE:
            # a duplicate is a fine frame, only surplus: dimmed less
            self._paint(state, theme.ACCENT_LEMON, 0.7)
        else:
            self._paint(state, theme.ACCENT_LIME, 1.0)

    def _parts_text(self):
        r = self.result
        text = "L %d · T %d · B %d" % (round(r.laplacian), round(r.tenengrad), round(r.brenner))
        # the shooting-condition adjustment is shown, not hidden
        offset = getattr(r, "expectation", 0.0) or 0.0
        if abs(offset) >= 0.5:
            text += "  ·  measured %d %s" % (
                round(r.raw_score), "(strict: good light)" if offset > 0 else "(lenient: hard light)"
            )
        if r.low_resolution:
            text += "  · low resolution"
        if r.source == raw_loader.SOURCE_RAW_DEMOSAIC:
            # a demosaiced RAW is not sharpened like a camera preview
            text += "  · demosaiced RAW"
        if r.override is True:
            text += "  ·  kept by hand"
        elif r.override is False:
            text += "  ·  rejected by hand"
        return text

    def _paint(self, state, color, image_opacity):
        score = "–" if self.result.score is None else "%d" % round(self.result.score)
        self.score_label.setText(
            '<span style="font-size:22px; font-weight:600; color:%s;">%s</span>'
            '<span style="font-size:11px; color:%s;">  %s</span>'
            % (color, score, theme.TEXT_SECONDARY, state)
        )
        # border through dynamic properties + one global stylesheet (no
        # per-card stylesheet parsing); manual decisions get a lemon frame
        self.setProperty("state", state)
        self.setProperty("manual", "true" if self.result.override is not None else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self._opacity.setOpacity(image_opacity)
