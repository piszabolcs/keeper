# Keeper - the author signature: a small footer button "psz // v{version}"
# that glows in the accent colour on hover and keyboard focus, and opens a
# compact "app info" card on click.

from PySide6.QtCore import QPropertyAnimation, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import branding, theme
from gui.scroll_fade import FadingScrollArea
from version import __version__

LABEL = "psz // v%s" % __version__

APP_NAME = "Keeper"
PURPOSE = (
    "Culls a folder of photos by sharpness: spots bursts and soft faces, moves "
    "rejects and duplicates aside (it never deletes) and saves corrected copies. "
    "Every run can be undone."
)
LAYERS = [
    ("core/", "Qt-free measuring and correction: sharpness, noise, EXIF, series, faces, tone"),
    ("utils/", "moving files, the write-ahead undo log, saving corrected copies"),
    ("gui/", "PySide6 window, background workers, preview and quick-check dialogs"),
    ("vendor/", "SCUNet network for the optional learned denoiser"),
]
BUILT_WITH = (
    "Python 3.12 · PySide6 (Qt 6) · OpenCV · NumPy · Pillow · rawpy / LibRaw · psutil · "
    "YuNet face model · optional PyTorch + SCUNet"
)
REPOSITORIES = [
    ("github.com/piszabolcs/Keeper", "https://github.com/piszabolcs/Keeper"),
    ("opencv/opencv_zoo", "https://github.com/opencv/opencv_zoo"),
    ("cszn/SCUNet", "https://github.com/cszn/SCUNet"),
    ("letmaik/rawpy · LibRaw/LibRaw", "https://github.com/letmaik/rawpy"),
]


def _animations_enabled():
    """Windows can switch UI animations off; then the glow appears instantly."""
    try:
        import ctypes

        enabled = ctypes.c_bool(True)
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0)
        return bool(enabled.value)
    except Exception:
        return True


class SignatureButton(QPushButton):
    """Bare, muted text button; accent glow on hover and keyboard focus."""

    GLOW_RADIUS = 24
    DURATION_MS = 180

    def __init__(self, parent=None):
        super().__init__(LABEL, parent)
        self.setObjectName("signature")
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setToolTip("About Keeper")
        self.setAccessibleName("About Keeper, version %s" % __version__)
        self._set_color(theme.TEXT_MUTED)

        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setColor(QColor(theme.ACCENT_LEMON))
        self._glow.setOffset(0, 0)
        self._glow.setBlurRadius(0)
        self.setGraphicsEffect(self._glow)

        self._animation = QPropertyAnimation(self._glow, b"blurRadius", self)
        self._animation.setDuration(self.DURATION_MS if _animations_enabled() else 0)
        self.clicked.connect(self.open_about)

    def _set_color(self, color):
        self.setStyleSheet(
            "QPushButton#signature { background: transparent; border: none;"
            " color: %s; font-size: 11px; padding: 4px 8px; }" % color
        )

    def _shine(self, on):
        self._animation.stop()
        self._animation.setStartValue(self._glow.blurRadius())
        self._animation.setEndValue(self.GLOW_RADIUS if on else 0)
        self._animation.start()
        self._set_color(theme.ACCENT_LEMON if on else theme.TEXT_MUTED)

    def enterEvent(self, event):
        self._shine(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.hasFocus():
            self._shine(False)
        super().leaveEvent(event)

    def focusInEvent(self, event):
        self._shine(True)
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if not self.underMouse():
            self._shine(False)
        super().focusOutEvent(event)

    def open_about(self):
        dialog = AboutDialog(self.window())
        dialog.setAttribute(Qt.WA_DeleteOnClose)  # no hidden dialogs pile up
        dialog.exec()


class AboutDialog(QDialog):
    """
    Frameless overlay over the whole parent window with a centred card;
    closes with the X button, Esc or a click outside the card. The card
    scrolls instead of being clipped on small windows.
    """

    CARD_WIDTH = 460
    MARK_HEIGHT = 36
    MARGIN = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        if parent is not None:
            self.setGeometry(parent.geometry())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        self.card = self._build_card()
        outer.addWidget(self.card, 0, Qt.AlignCenter)

    def _build_card(self):
        card = QWidget()
        card.setObjectName("dialogCard")
        card.setFixedWidth(self.CARD_WIDTH)
        card_layout = QVBoxLayout(card)
        # right/bottom room keeps the scrollbar off the rounded corners
        card_layout.setContentsMargins(0, 0, 6, 10)
        card_layout.setSpacing(0)

        # the header stays put; only the sections below it scroll
        self.header = QWidget()
        self.header.setStyleSheet("background: transparent;")
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(26, 20, 14, 10)
        header_layout.addLayout(self._header())
        card_layout.addWidget(self.header)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(26, 4, 14, 12)
        layout.setSpacing(12)

        self._section(layout, "What it does", PURPOSE)
        self._section(
            layout, "Architecture",
            "<br>".join("<b>%s</b> %s" % (name, text) for name, text in LAYERS),
        )
        self._section(layout, "Built with", BUILT_WITH)
        self._section(
            layout, "Repositories",
            "<br>".join(
                '<a href="%s" style="color:%s;">%s</a>' % (url, theme.ACCENT_LIME, name) if url else name
                for name, url in REPOSITORIES
            ),
        )

        self.scroll = FadingScrollArea(theme.DIALOG_FILL)
        self.scroll.setWidget(content)
        self.scroll.setWidgetResizable(True)
        card_layout.addWidget(self.scroll)
        return card

    def _header(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        mark = branding.mark_pixmap(self.MARK_HEIGHT)
        if mark is not None:
            label = QLabel()
            label.setPixmap(mark)
            label.setFixedSize(mark.size())
            row.addWidget(label)

        name = QLabel(APP_NAME)
        name.setStyleSheet("font-size: 20px; font-weight: 600; color: %s;" % theme.ACCENT_LEMON)
        row.addWidget(name)
        version = QLabel("v%s" % __version__)
        version.setObjectName("secondary")
        row.addWidget(version, 0, Qt.AlignBottom)
        row.addStretch()

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
        row.addWidget(close, 0, Qt.AlignTop)
        return row

    def _section(self, layout, title, text):
        section = QVBoxLayout()
        section.setSpacing(3)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 11px; font-weight: 600; color: %s;" % theme.ACCENT_LIME)
        section.addWidget(heading)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setOpenExternalLinks(True)
        body.setStyleSheet("font-size: 12px; color: %s;" % theme.TEXT_SECONDARY)
        section.addWidget(body)
        layout.addLayout(section)

    def resizeEvent(self, event):
        # the card is as tall as its content, but never taller than the window
        content = self.scroll.widget()
        content_width = self.CARD_WIDTH - 6
        content_height = (
            content.heightForWidth(content_width) if content.hasHeightForWidth() else content.sizeHint().height()
        ) + self.header.sizeHint().height() + 10 + 2
        self.card.setFixedHeight(min(content_height, max(160, self.height() - 2 * self.MARGIN)))
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 115))  # dimming layer
        painter.end()

    def mousePressEvent(self, event):
        if not self.card.geometry().contains(event.position().toPoint()):
            self.reject()
        else:
            super().mousePressEvent(event)
