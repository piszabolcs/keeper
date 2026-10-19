# Keeper - custom window frame.
#
# The native Windows caption does not fit a dark, tuned UI, so the frame is
# drawn by us - but moving and resizing are left to the OS by answering
# WM_NCHITTEST. That keeps Snap, Win+arrow, double-click maximise, the system
# menu and aero shake. (Calling startSystemMove() from a mouse event instead
# swallowed the button release in Windows' modal move loop, and the next click
# on any button was lost.)

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QWidget

from gui import branding, theme

TITLE_BAR_HEIGHT = 40
MARK_SIZE = 20
# border band grabbed for resizing; the content sits inside it
RESIZE_BORDER = 6

# Segoe Fluent / MDL2 glyphs, present on every Windows 10/11 machine
GLYPH_MINIMIZE = ""
GLYPH_MAXIMIZE = ""
GLYPH_RESTORE = ""
GLYPH_CLOSE = ""
ICON_FONT = "Segoe Fluent Icons, Segoe MDL2 Assets"

# the only red in the palette: a close button is universally red on hover
CLOSE_HOVER = "#C4443C"


def _button_style(hover_background, hover_text):
    return (
        "QPushButton { background: transparent; border: none; color: %s;"
        " font-family: %s; font-size: 10px; }"
        "QPushButton:hover { background: %s; color: %s; }"
        % (theme.TEXT_SECONDARY, ICON_FONT, hover_background, hover_text)
    )


class TitleBar(QWidget):
    """Logo, name and the three window buttons (Windows-sized, same place)."""

    BUTTON_WIDTH = 46

    def __init__(self, window):
        super().__init__(window)
        self.window_ = window
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        self.setObjectName("titleBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(10)

        mark = branding.mark_pixmap(MARK_SIZE)
        if mark is not None:
            label = QLabel()
            label.setPixmap(mark)
            label.setFixedSize(mark.size())
            layout.addWidget(label)

        name = QLabel(window.windowTitle() or "Keeper")
        name.setStyleSheet("font-size: 12px; font-weight: 600; color: %s;" % theme.TEXT_SECONDARY)
        layout.addWidget(name)
        layout.addStretch()

        self.minimize_button = self._button(GLYPH_MINIMIZE, theme.GLASS_FILL_HOVER, "Minimize")
        self.minimize_button.clicked.connect(window.showMinimized)
        layout.addWidget(self.minimize_button)

        self.maximize_button = self._button(GLYPH_MAXIMIZE, theme.GLASS_FILL_HOVER, "Maximize")
        self.maximize_button.clicked.connect(self.toggle_maximized)
        layout.addWidget(self.maximize_button)

        self.close_button = self._button(GLYPH_CLOSE, CLOSE_HOVER, "Close")
        self.close_button.clicked.connect(window.close)
        layout.addWidget(self.close_button)

    def _button(self, glyph, hover_background, name):
        button = QPushButton(glyph)
        button.setFixedSize(self.BUTTON_WIDTH, TITLE_BAR_HEIGHT)
        button.setCursor(Qt.ArrowCursor)
        button.setFocusPolicy(Qt.NoFocus)
        button.setAccessibleName(name)
        button.setToolTip(name)
        button.setStyleSheet(_button_style(hover_background, theme.TEXT_PRIMARY))
        return button

    def toggle_maximized(self):
        if self.window_.isMaximized():
            self.window_.showNormal()
        else:
            self.window_.showMaximized()
        self.update_glyph()

    def update_glyph(self):
        """Maximised windows show the "restore" glyph."""
        self.maximize_button.setText(GLYPH_RESTORE if self.window_.isMaximized() else GLYPH_MAXIMIZE)

    def is_draggable(self, point):
        """True where the bar can be grabbed (everything except the buttons)."""
        return not isinstance(self.childAt(point), QPushButton)


# WM_NCHITTEST and the answers used to tell Windows what the mouse is over
WM_NCHITTEST = 0x0084
HTCLIENT = 1
HTCAPTION = 2
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17


class FramelessWindow(QMainWindow):
    """
    Frameless main window that still resizes at its edges. The edges are
    reported through WM_NCHITTEST (which arrives before any child widget can
    swallow the mouse), so the OS does the resizing with native cursors and
    snapping.
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self._caption_bar = None

    def set_caption_bar(self, bar):
        """The widget that acts as the caption (drag handle)."""
        self._caption_bar = bar

    def _hit_test(self, global_point):
        """
        Windows hit-test code for a point in Qt's LOGICAL global coordinates.
        (The message itself carries physical pixels; nativeEvent therefore
        asks QCursor, which is DPI-correct on scaled and mixed-DPI screens.)
        """
        pos = self.mapFromGlobal(global_point)

        # resizing only makes sense for a normal (not maximised) window
        if not self.isMaximized():
            left = pos.x() <= RESIZE_BORDER
            right = pos.x() >= self.width() - RESIZE_BORDER
            top = pos.y() <= RESIZE_BORDER
            bottom = pos.y() >= self.height() - RESIZE_BORDER

            for hit, code in (
                (top and left, HTTOPLEFT), (top and right, HTTOPRIGHT),
                (bottom and left, HTBOTTOMLEFT), (bottom and right, HTBOTTOMRIGHT),
                (left, HTLEFT), (right, HTRIGHT), (top, HTTOP), (bottom, HTBOTTOM),
            ):
                if hit:
                    return code

        # the caption is a handle even when maximised (drag down to restore)
        if self._caption_bar is not None:
            bar_point = self._caption_bar.mapFrom(self, pos)
            if self._caption_bar.rect().contains(bar_point) and self._caption_bar.is_draggable(bar_point):
                return HTCAPTION

        return HTCLIENT

    def nativeEvent(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            try:
                import ctypes.wintypes

                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == WM_NCHITTEST:
                    zone = self._hit_test(QCursor.pos())
                    if zone != HTCLIENT:
                        return True, zone
            except Exception:
                # anything unexpected: default behaviour (no edge resize)
                pass
        return super().nativeEvent(event_type, message)


def apply_rounded_corners(window):
    """Windows 11 rounded corners for the frameless window; ignored elsewhere."""
    try:
        import ctypes

        preference = ctypes.c_int(2)  # DWMWCP_ROUND
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            int(window.winId()), 33, ctypes.byref(preference), ctypes.sizeof(preference)
        )
    except Exception:
        pass


def point(x, y):
    """Small helper for tests: a QPoint from ints."""
    return QPoint(int(x), int(y))
