# Keeper - icon and logo. Loading is lenient everywhere: a missing branding
# file means no icon, never a crash.

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ICON_FILE = ASSETS / "keeper.ico"
MARK_FILE = ASSETS / "logo_mark.png"

# Windows groups taskbar buttons by this id (see register_windows_app_id)
APP_USER_MODEL_ID = "psz.Keeper.PhotoAssistant"


def app_icon():
    """The application icon, or an empty icon if the file is missing."""
    return QIcon(str(ICON_FILE)) if ICON_FILE.is_file() else QIcon()


def mark_pixmap(height):
    """The aperture mark scaled to height, or None."""
    if not MARK_FILE.is_file():
        return None
    image = QPixmap(str(MARK_FILE))
    if image.isNull():
        return None
    return image.scaledToHeight(height, Qt.SmoothTransformation)


def register_windows_app_id():
    """
    Tells Windows this is an application of its own; otherwise the taskbar
    shows the Python icon and groups the window with every pythonw.exe. Must
    run BEFORE the QApplication exists. Skipped silently elsewhere.
    """
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass
