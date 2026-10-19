# Keeper - background measuring.
#
# A QThread drives a thread pool: OpenCV and JPEG decoding release the GIL,
# so threads really run in parallel without copying images between
# processes. Thumbnails are decoded here too, so the GUI thread never decodes.

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QThread, Signal

import config
from core.sharpness import SharpnessResult, measure_sharpness
from gui.result_card import thumbnail_image

log = logging.getLogger(__name__)


def measure_item(item):
    """Measures one shot and attaches its thumbnail; never raises."""
    try:
        result = measure_sharpness(item.primary, item.files)
    except Exception as exc:  # measure_sharpness should not raise; be safe anyway
        log.exception("measuring %s failed", item.primary)
        return SharpnessResult(
            path=item.primary, score=None, laplacian=None, tenengrad=None, brenner=None,
            low_resolution=False, error=str(exc), files=item.files,
        )
    if result.error:
        log.warning("cannot read %s: %s", result.path, result.error)
    try:
        result.thumbnail = thumbnail_image(result.path)
    except Exception:
        result.thumbnail = None
    return result


class SharpnessWorker(QThread):
    # done, total, SharpnessResult
    progress = Signal(int, int, object)
    # all results, cancelled?
    finished_all = Signal(list, bool)

    def __init__(self, items, folder, parent=None):
        # items are ImageItems (one SHOT each, possibly two files); folder is
        # the folder they belong to - results stay bound to it
        super().__init__(parent)
        self.items = list(items)
        self.folder = folder
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        results = []
        total = len(self.items)
        try:
            pool = ThreadPoolExecutor(max_workers=config.WORKER_THREADS)
            futures = [pool.submit(measure_item, item) for item in self.items]
            try:
                # submission order, not completion order: the grid follows the
                # file names (shooting order) instead of jumping around
                for future in futures:
                    if self._cancel.is_set():
                        break
                    results.append(future.result())
                    self.progress.emit(len(results), total, results[-1])
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.exception("the measuring run failed")
            self._cancel.set()  # an incomplete list must not be treated as complete
        self.finished_all.emit(results, self._cancel.is_set())
