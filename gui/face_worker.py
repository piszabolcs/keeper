# Keeper - background face analysis. A separate pass, not part of measuring:
# it needs the model file, is slower, and needs colour pixels while measuring
# loads grey ones. Every analyse_path call may run on a different pool thread
# (core/faces.py keeps one detector per thread for that reason).

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QThread, Signal

import config
from core import faces

log = logging.getLogger(__name__)


class FaceWorker(QThread):
    # done, total, SharpnessResult (its faces field filled in)
    progress = Signal(int, int, object)
    # shots with a face, cancelled?
    finished_all = Signal(int, bool)

    def __init__(self, results, parent=None):
        super().__init__(parent)
        # unreadable shots cannot be opened for faces either
        self.results = [r for r in results if r.score is not None]
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        total, with_faces, done = len(self.results), 0, 0
        try:
            pool = ThreadPoolExecutor(max_workers=config.WORKER_THREADS)
            futures = [pool.submit(faces.analyse_path, r.path) for r in self.results]
            try:
                for result, future in zip(self.results, futures):
                    if self._cancel.is_set():
                        break
                    result.faces = future.result()
                    with_faces += 1 if result.faces.count else 0
                    done += 1
                    self.progress.emit(done, total, result)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.exception("the face analysis failed")
            self._cancel.set()
        self.finished_all.emit(with_faces, self._cancel.is_set())
