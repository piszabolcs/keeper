# Keeper - entry point.

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from version import __version__

log = logging.getLogger("keeper")


def log_directory():
    """%LOCALAPPDATA%\\Keeper\\logs on Windows, ~/.local/share/Keeper/logs elsewhere."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "Keeper" / "logs"


def setup_logging():
    """
    A rotating log file (3 x 1 MB). Under pythonw there is no console, so
    without this every traceback would be lost; unhandled exceptions are
    logged too. Paths are logged, image content never.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        folder = log_directory()
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            folder / "keeper.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    except OSError:
        pass  # no log file is no reason not to start
    if sys.stderr is not None:  # console mode (python.exe, not pythonw.exe)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    def unhandled(kind, value, traceback):
        log.critical("unhandled exception", exc_info=(kind, value, traceback))

    sys.excepthook = unhandled


def main():
    setup_logging()
    log.info("Keeper %s starting", __version__)

    from gui import branding, frame, theme
    from gui.main_window import MainWindow

    # the taskbar id must be set BEFORE the QApplication exists
    branding.register_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName("Keeper")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(branding.app_icon())
    app.setStyleSheet(theme.stylesheet())

    window = MainWindow()
    window.show()
    # native Windows touches need a real window handle, i.e. after show()
    theme.apply_windows_chrome(window)
    frame.apply_rounded_corners(window)

    # "Keeper.bat <folder or photo>" / "Open with": a folder is loaded, a
    # single photo opens the quick check
    if len(sys.argv) > 1:
        QTimer.singleShot(0, lambda: window.open_path(sys.argv[1]))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
