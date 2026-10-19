# Keeper - regenerates the README screenshots.
#
#   .venv\Scripts\python.exe tools\screenshots.py [output folder]
#
# Everything is rendered OFFSCREEN (no window ever appears) from SYNTHETIC
# photos drawn here: no real photo, person, path or user name can end up in
# an image. The default output is assets/screenshots/. The script fails if
# any expected image is missing.

import os
import pathlib
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    # the offscreen platform finds no fonts on its own
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtGui import QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# JPEG keeps the noisy synthetic photos small; the UI text stays crisp at q90
EXPECTED = ["main_window.jpg", "threshold.jpg", "raw_pairs.jpg", "preview.jpg", "about.jpg", "quick_check.jpg"]
SHOWN_FOLDER = r"D:\Photos\2026-07-27 Lake trip"   # invented, never a real path
SIZE = (1400, 1050)
START = 1_785_000_000  # capture times (EXIF) of the synthetic shots


def _pink(rng, height, width, alpha):
    """1/f noise: natural images have roughly this spectrum."""
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spectrum = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) / f ** alpha
    spectrum[0, 0] = 0
    out = np.fft.irfft2(spectrum, s=(height, width)).astype(np.float32)
    return out / out.std()


def synthetic_photo(seed, blur, shift=(0, 0)):
    """An abstract, photo-like frame; blur sets how soft it is."""
    rng = np.random.default_rng(seed)
    height, width = SIZE[1], SIZE[0]
    light, colour = _pink(rng, height, width, 1.25), _pink(rng, height, width, 1.55)
    tint = rng.uniform(-30, 30, 3)
    image = np.empty((height, width, 3), np.float32)
    for channel in range(3):
        image[:, :, channel] = 120 + tint[channel] + 45 * light + 16 * colour * (channel - 1)
    if shift != (0, 0):
        image = cv2.warpAffine(image, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                               (width, height), borderMode=cv2.BORDER_REFLECT)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    # in-camera sharpening, like every camera JPEG
    image = image + 2.5 * (image - cv2.GaussianBlur(image, (0, 0), 0.8))
    return np.clip(image, 0, 255).astype(np.uint8)


def save_jpeg(path, rgb, captured, iso=200, exposure=1 / 250.0):
    image = Image.fromarray(rgb)
    exif = image.getexif()
    details = exif.get_ifd(0x8769)
    details[0x9003] = __import__("time").strftime("%Y:%m:%d %H:%M:%S", __import__("time").localtime(captured))
    details[0x8827] = iso
    details[0x829A] = exposure
    details[0x920A] = 50.0
    image.save(path, quality=95, subsampling=0, exif=exif)


def make_folder(folder):
    """Twelve single shots of decreasing sharpness and a four-frame burst."""
    captured = START
    blurs = [0.0, 0.2, 0.4, 0.5, 1.2, 0.6, 0.7, 1.6, 0.3, 0.9, 1.1, 0.8]
    for index, blur in enumerate(blurs):
        captured += 240
        iso = 6400 if index == 6 else 200
        exposure = 1 / 15.0 if index == 7 else 1 / 250.0
        save_jpeg(folder / ("DSC0%03d.jpg" % (100 + index)), synthetic_photo(index, blur), captured, iso, exposure)

    for index, blur in enumerate([0.9, 0.0, 0.3, 1.4]):
        captured += 1
        save_jpeg(folder / ("DSC0%03d.jpg" % (200 + index)), synthetic_photo(99, blur, (index * 4, -index * 3)), captured)


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication([])
    from gui import theme
    from gui.main_window import MainWindow
    from gui.preview_dialog import PreviewDialog
    from gui.signature import AboutDialog

    app.setStyleSheet(theme.stylesheet())

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def measure(window, folder):
        window.set_folder(folder)
        window.folder_label.setText(SHOWN_FOLDER)
        window.folder_label.setToolTip(SHOWN_FOLDER)
        window._start_measuring()
        loop = QEventLoop()
        window.worker.finished_all.connect(lambda results, cancelled: QTimer.singleShot(0, loop.quit))
        QTimer.singleShot(60000, loop.quit)   # never hang
        loop.exec()
        app.processEvents()

    def save(widget, name, overlay=None):
        image = widget.grab()
        if overlay is not None:
            painter = QPainter(image)
            painter.drawPixmap(0, 0, overlay.grab())
            painter.end()
        image.save(str(out / name), "JPG", 90)
        print("%-18s %d x %d" % (name, image.width(), image.height()))

    work = pathlib.Path(tempfile.mkdtemp(prefix="keeper_screens_"))
    try:
        photos = work / "photos"
        photos.mkdir()
        make_folder(photos)

        window = MainWindow()
        window.resize(1280, 820)
        window.show()
        measure(window, photos)
        window.threshold_slider.setValue(50)
        window._detect_series()
        from core import series

        series.keep_best_n(window.series, 1)
        window._recolor_cards()
        app.processEvents()
        save(window, "main_window.jpg")

        window.threshold_slider.setValue(70)
        app.processEvents()
        save(window, "threshold.jpg")
        window.threshold_slider.setValue(50)

        sample = window._sample_result()
        dialog = PreviewDialog(sample, 150, 150, 100, 0, False, 100, window)
        dialog.show()
        for _ in range(100):   # loading + first render
            wait(100)
            if dialog.after_label.pixmap() is not None and not dialog.after_label.pixmap().isNull():
                break
        app.processEvents()
        save(window, "preview.jpg", overlay=dialog)
        dialog.close()

        about = AboutDialog(window)
        about.show()
        wait(300)
        save(window, "about.jpg", overlay=about)
        about.close()

        from gui.quick_check import QuickCheckDialog

        check = QuickCheckDialog(photos / "DSC0106.jpg", 50, window._strength_values(), False, window)
        check.show()
        for _ in range(100):
            wait(100)
            if check.photo is not None:
                break
        wait(200)
        save(window, "quick_check.jpg", overlay=check)
        check.close()

        # RAW+JPEG pairs: empty .ARW companions, pairing goes by the file stem
        pairs = work / "pairs"
        pairs.mkdir()
        for index, source in enumerate(sorted(photos.glob("DSC01*.jpg"))[:8]):
            shutil.copy2(source, pairs / source.name)
            if index % 2 == 0:
                (pairs / (source.stem + ".ARW")).write_bytes(b"")
        other = MainWindow()
        other.resize(1280, 820)
        other.show()
        measure(other, pairs)
        app.processEvents()
        save(other, "raw_pairs.jpg")
        other.close()
        window.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    missing = [name for name in EXPECTED if not (out / name).is_file()]
    if missing:
        sys.exit("missing screenshots: %s" % ", ".join(missing))
    print("done - images in %s" % out)


if __name__ == "__main__":
    main()
