# Keeper - collapsible panel section. The left column grew to about two
# screens; the user decides what stays open instead of scrolling past
# controls not needed right now.

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from gui import theme

# chevron right / chevron down - the familiar file-tree convention
GLYPH_CLOSED = ""
GLYPH_OPEN = ""
ICON_FONT = "Segoe Fluent Icons, Segoe MDL2 Assets"


class CollapsibleSection(QFrame):
    """A glass panel with a clickable header; fill content_layout like a QVBoxLayout."""

    def __init__(self, title, open_=True, parent=None):
        super().__init__(parent)
        self.setObjectName("glassPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        # never shrink below the natural height: inside a scroll area Qt would
        # rather squash the buttons to a few pixels than show a scrollbar
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 14, 20, 16)
        outer.setSpacing(10)

        self.header = QPushButton()
        self.header.setObjectName("sectionHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setAccessibleName(title)
        self.header.clicked.connect(self.toggle)

        # title and chevron are labels that let the mouse through to the button
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.chevron = QLabel()
        self.chevron.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._highlight(False)
        header_layout.addWidget(self.chevron)
        outer.addWidget(self.header)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        outer.addWidget(self.content)

        self.set_open(open_)

    # hover feedback comes from code: a "header:hover QLabel" stylesheet rule
    # is always true for the child label in Qt
    def enterEvent(self, event):
        self._highlight(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._highlight(False)
        super().leaveEvent(event)

    def _highlight(self, on):
        self.chevron.setStyleSheet(
            "font-family: %s; font-size: 11px; color: %s;"
            % (ICON_FONT, theme.ACCENT_LIME if on else theme.TEXT_MUTED)
        )

    def set_open(self, open_):
        self.is_open = bool(open_)
        self.content.setVisible(self.is_open)
        self.chevron.setText(GLYPH_OPEN if self.is_open else GLYPH_CLOSED)
        # a closed section needs no bottom margin
        self.layout().setContentsMargins(20, 14, 20, 16 if self.is_open else 4)

    def toggle(self):
        self.set_open(not self.is_open)
