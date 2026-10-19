# Keeper - look and feel in one place: colours, stylesheet, background.
#
# The base is a colour-neutral dark grey on purpose: in a photo culling tool
# a tinted UI skews the eye's judgement of exposure and colour. Lemon and lime
# appear only as accents, so the photos stay the main actors.

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import QGraphicsBlurEffect, QGraphicsPixmapItem, QGraphicsScene

# --- palette ---

BG_TOP = "#111214"
BG_BOTTOM = "#17181B"

GLASS_FILL = "rgba(255, 255, 255, 0.06)"
GLASS_FILL_HOVER = "rgba(255, 255, 255, 0.10)"
GLASS_BORDER = "rgba(255, 255, 255, 0.12)"
DIALOG_FILL = "#1B1C20"

ACCENT_LEMON = "#FFE94A"
ACCENT_LIME = "#C6F24E"
BUTTON_TEXT = "#14150F"

TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "rgba(255, 255, 255, 0.62)"
# 48 % white on the panel background is about 4.6:1 (WCAG AA for small text)
TEXT_MUTED = "rgba(255, 255, 255, 0.48)"

CARD_RADIUS = 16
PANEL_WIDTH = 300

FONT_FAMILY = "Segoe UI Variable, Segoe UI, sans-serif"


def stylesheet():
    """The application-wide stylesheet."""
    return """
    QWidget {
        color: %(text)s;
        font-family: %(font)s;
        font-size: 13px;
    }

    /* glass panels and cards */
    QFrame#glassPanel, QFrame#resultCard {
        background: %(glass)s;
        border: 1px solid %(border)s;
        border-radius: %(radius)dpx;
    }
    /* result cards are restyled through dynamic properties (no per-card
       stylesheet parsing on every threshold step) */
    QFrame#resultCard[state="kept"] { border: 1px solid %(lime)s; }
    QFrame#resultCard[manual="true"] { border: 2px solid %(lemon)s; }

    QWidget#dialogCard {
        background: %(dialog)s;
        border: 1px solid %(border)s;
        border-radius: %(radius)dpx;
    }

    QLabel#panelTitle { font-size: 15px; font-weight: 600; color: %(text)s; }
    QLabel#secondary { color: %(secondary)s; }
    QLabel#muted { color: %(muted)s; font-size: 11px; }
    QLabel#dropHint { color: %(muted)s; font-size: 15px; }

    /* primary button: lemon -> lime gradient with dark text. min-height
       leaves room for descenders (g, j, p), otherwise they are clipped. */
    QPushButton#primary {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 %(lemon)s, stop:1 %(lime)s);
        color: %(buttonText)s;
        border: none;
        border-radius: 10px;
        padding: 11px 16px;
        min-height: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    QPushButton#primary:hover   { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FFEF6E, stop:1 #D3F76B); }
    QPushButton#primary:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E8D340, stop:1 #B2DC44); }
    QPushButton#primary:disabled { background: rgba(255, 255, 255, 0.08); color: %(muted)s; }

    /* collapsible section header: not a command, so it does not look like a
       button; its hover feedback is set from code (see section.py) */
    QPushButton#sectionHeader { background: transparent; border: none; padding: 0; text-align: left; }

    /* secondary button: glass */
    QPushButton#secondary {
        background: %(glass)s;
        color: %(text)s;
        border: 1px solid %(border)s;
        border-radius: 10px;
        padding: 10px 16px;
        min-height: 20px;
    }
    QPushButton#secondary:hover    { background: %(glassHover)s; }
    QPushButton#secondary:disabled { color: %(muted)s; border-color: rgba(255,255,255,0.06); }

    QSlider::groove:horizontal { height: 4px; background: rgba(255, 255, 255, 0.14); border-radius: 2px; }
    QSlider::sub-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 %(lemon)s, stop:1 %(lime)s);
        border-radius: 2px;
    }
    QSlider::handle:horizontal { width: 16px; height: 16px; margin: -7px 0; border-radius: 8px; background: %(lemon)s; }

    QProgressBar {
        background: rgba(255, 255, 255, 0.08); border: none; border-radius: 3px;
        height: 6px; text-align: center; color: transparent;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 %(lemon)s, stop:1 %(lime)s);
        border-radius: 3px;
    }

    QListWidget { background: transparent; border: none; outline: none; }
    QListWidget::item { border: none; }
    QListWidget::item:selected { background: transparent; }

    QScrollArea { background: transparent; border: none; }
    QScrollBar:vertical { background: transparent; width: 10px; margin: 10px 0; }
    QScrollBar::handle:vertical { background: rgba(255, 255, 255, 0.16); border-radius: 5px; min-height: 28px; }
    QScrollBar::handle:vertical:hover { background: rgba(255, 255, 255, 0.26); }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
    """ % {
        "text": TEXT_PRIMARY,
        "secondary": TEXT_SECONDARY,
        "muted": TEXT_MUTED,
        "glass": GLASS_FILL,
        "glassHover": GLASS_FILL_HOVER,
        "border": GLASS_BORDER,
        "dialog": DIALOG_FILL,
        "lemon": ACCENT_LEMON,
        "lime": ACCENT_LIME,
        "buttonText": BUTTON_TEXT,
        "radius": CARD_RADIUS,
        "font": FONT_FAMILY,
    }


# The blurred background is drawn ONCE at a generous fixed size and then
# stretched to the window; the stretch is invisible on a blurred image and
# resizing does not re-run an expensive blur every frame.
BACKGROUND_WIDTH = 1920
BACKGROUND_HEIGHT = 1200

_background = None


def background_pixmap():
    """The blurred background, built on first use and cached."""
    global _background
    if _background is None:
        _background = build_background(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    return _background


def build_background(width, height):
    """Dark gradient with two faint accent glows, blurred (Qt has no backdrop-filter)."""
    width, height = max(1, width), max(1, height)
    base = QPixmap(width, height)
    base.fill(Qt.transparent)

    painter = QPainter(base)
    painter.setRenderHint(QPainter.Antialiasing, True)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QColor(BG_TOP))
    gradient.setColorAt(1.0, QColor(BG_BOTTOM))
    painter.fillRect(0, 0, width, height, gradient)
    _glow(painter, width * 0.18, height * 0.12, max(width, height) * 0.55, QColor(255, 233, 74, 26))
    _glow(painter, width * 0.88, height * 0.85, max(width, height) * 0.50, QColor(198, 242, 78, 20))
    painter.end()

    return _blur(base, 90)


def _glow(painter, x, y, radius, color):
    gradient = QRadialGradient(x, y, radius)
    gradient.setColorAt(0.0, color)
    gradient.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
    painter.setBrush(gradient)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))


def _blur(pixmap, radius):
    scene = QGraphicsScene()
    item = QGraphicsPixmapItem(pixmap)
    effect = QGraphicsBlurEffect()
    effect.setBlurRadius(radius)
    item.setGraphicsEffect(effect)
    scene.addItem(item)

    output = QImage(pixmap.size(), QImage.Format_ARGB32_Premultiplied)
    output.fill(QColor(BG_TOP))
    painter = QPainter(output)
    painter.setRenderHint(QPainter.Antialiasing, True)
    scene.render(painter)
    painter.end()
    return QPixmap.fromImage(output)


def apply_windows_chrome(window):
    """Windows 11 dark title bar and acrylic backdrop; silently skipped elsewhere."""
    try:
        import ctypes

        hwnd = int(window.winId())
        dark = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark))
        backdrop = ctypes.c_int(3)  # DWMSBT_TRANSIENTWINDOW
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
    except Exception:
        pass
