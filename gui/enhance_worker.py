# Keeper - background correction.
#
# Unlike measuring this runs the images ONE AFTER ANOTHER: OpenCV's NLM
# already uses every core, extra threads only fight each other (measured on
# four 6 MP frames: 5.8 s sequential vs 14.9 s in parallel).

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from core.enhance import Cancelled, enhance_image, load_bgr_full
from core.noise import read_iso, sigma_from_gray
from utils.save_ops import SaveReport, save_corrected_batch

log = logging.getLogger(__name__)


@dataclass
class EnhanceProgress:
    """What happened to one image, for the status line."""

    path: Path
    sigma: float | None = None
    denoise_h: float = 0.0
    sharpen_amount: float = 0.0
    gains: tuple = (1.0, 1.0, 1.0)
    gamma: float = 1.0
    # local contrast and saturation MUST be reported: on flat, dull frames
    # they are the only steps that act
    clahe: float = 0.0
    saturation: float = 1.0
    learned: bool = False
    error: str | None = None

    # the same four conditions as core.enhance.EnhanceResult - the preview
    # shows that one, the batch this one; they must give the same answer
    @property
    def tone_changed(self):
        return (
            any(abs(g - 1.0) > 1e-6 for g in self.gains)
            or abs(self.gamma - 1.0) > 1e-6
            or self.clahe > 0.0
            or abs(self.saturation - 1.0) > 1e-6
        )

    @property
    def anything_changed(self):
        return self.denoise_h > 0.0 or self.sharpen_amount > 0.0 or self.tone_changed


class EnhanceWorker(QThread):
    # done, total, EnhanceProgress
    progress = Signal(int, int, object)
    # SaveReport, cancelled?
    finished_all = Signal(object, bool)

    def __init__(self, folder, results, denoise_percent=100, sharpen_percent=100,
                 brightness_percent=100, wb_percent=0, learned=False, color_percent=100, parent=None):
        # results: only the KEPT shots; the percentages scale the automatic strengths
        super().__init__(parent)
        self.folder = folder
        self.results = list(results)
        self.settings = dict(
            denoise_percent=denoise_percent, sharpen_percent=sharpen_percent,
            brightness_percent=brightness_percent, wb_percent=wb_percent,
            learned=learned, color_percent=color_percent,
        )
        self.errors = []
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _items(self):
        """
        A generator, not a list: a 24 MP image is 72 MB raw; producing all of
        them before saving would need tens of GB. One image at a time.
        """
        total = len(self.results)
        for index, result in enumerate(self.results, start=1):
            if self._cancel.is_set():
                return
            try:
                # the full-size image is loaded once; noise is measured from it
                bgr = load_bgr_full(result.path)
                sigma = sigma_from_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                corrected = enhance_image(
                    bgr, sigma, result.score, iso=read_iso(result.path),
                    cancel=self._cancel, **self.settings,
                )
            except Cancelled:
                return
            except Exception as exc:
                # one bad file must not stop the batch
                log.warning("correcting %s failed: %s", result.path, exc)
                self.errors.append((result.path, str(exc)))
                self.progress.emit(index, total, EnhanceProgress(path=result.path, error=str(exc)))
                continue

            self.progress.emit(index, total, EnhanceProgress(
                path=result.path, sigma=corrected.sigma, denoise_h=corrected.denoise_h,
                sharpen_amount=corrected.sharpen_amount, gains=corrected.gains,
                gamma=corrected.gamma, clahe=corrected.clahe,
                saturation=corrected.saturation, learned=corrected.learned,
            ))
            yield result.path, corrected.image

    def run(self):
        try:
            report = save_corrected_batch(self.folder, self._items())
        except Exception as exc:
            # e.g. the undo log could not be written: report it, never hang the UI
            log.exception("saving corrected copies failed")
            report = SaveReport(error=str(exc))
        report.failed.extend(self.errors)
        self.finished_all.emit(report, self._cancel.is_set())
