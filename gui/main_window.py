# Keeper - the main window.
#
# The main button works in two steps on purpose: starting only MEASURES.
# Afterwards the threshold slider recolours the grid live (the scores exist),
# and files move only on the second click - the user always sees beforehand
# what will happen.
#
# One state machine drives every control (_refresh_state). While a run is
# active (measuring, correcting, face analysis) nothing that could change the
# folder or its files is enabled, and results stay bound to the folder they
# were measured in.

import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import config
from core import faces, learned_denoise, series
from core.enhance import scaled_strengths
from core.sharpness import clear_overrides, is_kept
from gui import theme
from gui.scroll_fade import FadingScrollArea
from gui.enhance_worker import EnhanceWorker
from gui.face_worker import FaceWorker
from gui.frame import RESIZE_BORDER, FramelessWindow, TitleBar
from gui.preview_dialog import WB_TOOLTIP, PreviewDialog
from gui.quick_check import QuickCheckDialog
from gui.result_card import CARD_WIDTH, ResultCard
from gui.section import CollapsibleSection
from gui.signature import SignatureButton
from gui.status_panel import (
    ACTIVITY_ENHANCING,
    ACTIVITY_FACES,
    ACTIVITY_MEASURING,
    StatusPanel,
    elided,
)
from gui.worker import SharpnessWorker
from utils.file_ops import (
    KIND_CORRECTION,
    collect_images,
    destination_folder,
    has_undoable_run,
    last_run_summary,
    move_to_rejects,
    undo_last_run,
)
from utils.save_ops import undo_last_correction

log = logging.getLogger(__name__)

STATE_IDLE = "idle"            # nothing measured yet: can start
STATE_MEASURING = "measuring"  # can cancel
STATE_MEASURED = "measured"    # can move, correct, analyse faces
STATE_ENHANCING = "enhancing"  # can cancel
STATE_FACES = "faces"          # can cancel
BUSY_STATES = (STATE_MEASURING, STATE_ENHANCING, STATE_FACES)

FOLDER_LABEL_WIDTH = theme.PANEL_WIDTH - 40


def shots_text(count, where):
    """"shot", not "file": a RAW+JPEG pair is two files but one photo."""
    return "%d shot%s %s" % (count, "" if count == 1 else "s", where)


class MainWindow(FramelessWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keeper")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)

        self.folder = None
        self.results = []
        self.cards = []
        # list items next to the cards: series labels make cards taller, so
        # the cell sizes have to be refreshed afterwards
        self.grid_items = []
        self.series = []
        self.worker = None
        self.enhance_worker = None
        self.face_worker = None
        self.state = STATE_IDLE
        # reading the log on every slider step was measurable; cached and
        # refreshed after every operation that can change it
        self._undo_available = False
        self._closing = False

        self._build()
        # a folder or a single photo can be dropped anywhere on the window
        self.setAcceptDrops(True)
        self._refresh_state()
        self.status_panel.start_monitoring()

    # -----------------------------------------------------------------
    # building
    # -----------------------------------------------------------------

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        # the outermost band belongs to resizing (Windows grabs the edge there)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(RESIZE_BORDER, RESIZE_BORDER, RESIZE_BORDER, RESIZE_BORDER)
        outer.setSpacing(0)

        self.title_bar = TitleBar(self)
        self.set_caption_bar(self.title_bar)
        outer.addWidget(self.title_bar)

        column = QVBoxLayout()
        column.setContentsMargins(12, 12, 12, 4)
        column.setSpacing(12)
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_left_column())
        body.addWidget(self._build_grid(), 1)
        column.addLayout(body, 1)
        column.addWidget(self._build_footer())
        outer.addLayout(column, 1)

    def _build_left_column(self):
        """
        Controls in a SCROLL area (on a low window Qt would otherwise squash
        the buttons) and the status panel fixed below it - during a run that
        is exactly what the user watches.
        """
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_sort_section())
        layout.addWidget(self._build_series_section())
        layout.addWidget(self._build_faces_section())
        self.enhance_section = self._build_enhance_section()
        layout.addWidget(self.enhance_section)
        layout.addWidget(self._build_undo_button())
        layout.addStretch()

        scroll = FadingScrollArea(theme.BG_TOP, theme.BG_BOTTOM)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)

        self.status_panel = StatusPanel()
        holder = QWidget()
        holder.setFixedWidth(theme.PANEL_WIDTH + 14)
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(12)
        holder_layout.addWidget(scroll, 1)
        holder_layout.addWidget(self.status_panel)
        return holder

    def _button(self, text, slot, name="secondary", tooltip=None):
        button = QPushButton(text)
        button.setObjectName(name)
        button.setCursor(Qt.PointingHandCursor)
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(slot)
        return button

    def _label(self, name="muted", wrap=True):
        label = QLabel()
        label.setObjectName(name)
        label.setWordWrap(wrap)
        return label

    def _build_sort_section(self):
        section = CollapsibleSection("Focus sorting", open_=True)
        layout = section.content_layout

        self.folder_button = self._button("Choose folder…", self._choose_folder)
        layout.addWidget(self.folder_button)
        self.photo_button = self._button(
            "Check one photo…", self._check_photo,
            tooltip="Instant verdict and correction for a single photo.\nNothing is moved or logged.",
        )
        layout.addWidget(self.photo_button)
        self.folder_label = self._label(wrap=False)
        self.folder_label.setText("No folder chosen")
        layout.addWidget(self.folder_label)
        layout.addSpacing(6)

        self.threshold_label = QLabel()
        layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(config.DEFAULT_THRESHOLD)
        self.threshold_slider.setAccessibleName("Sharpness threshold")
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_slider)
        hint = self._label()
        hint.setText("Shots scoring below this go to the %s folder." % config.FOLDER_REJECTS)
        layout.addWidget(hint)
        layout.addSpacing(10)

        self.main_button = self._button("", self._on_main_button, name="primary")
        layout.addWidget(self.main_button)
        layout.addSpacing(4)

        # hidden while empty: an empty label would still take a line
        self.summary_label = self._label("secondary")
        self.summary_label.setVisible(False)
        layout.addWidget(self.summary_label)
        return section

    def _build_series_section(self):
        """The threshold knows nothing about bursts: it keeps or drops them all."""
        section = CollapsibleSection("Series", open_=True)
        layout = section.content_layout

        self.series_label = self._label()
        layout.addWidget(self.series_label)
        self.protect_button = self._button(
            "Protect the best of each series", self._protect_series_best,
            tooltip="The best frame of every series is kept even below the\nthreshold. Other frames are not touched.",
        )
        layout.addWidget(self.protect_button)

        # how many to keep depends on the moment: one for a calm landscape,
        # two or three for a running child
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Keep per series:"))
        self.keep_spin = QSpinBox()
        self.keep_spin.setRange(1, config.SERIES_KEEP_MAX)
        self.keep_spin.setValue(config.SERIES_KEEP_DEFAULT)
        self.keep_spin.setAccessibleName("Shots to keep per series")
        row.addWidget(self.keep_spin)
        row.addStretch(1)
        layout.addLayout(row)

        self.dedupe_button = self._button(
            "Mark duplicates", self._filter_duplicates,
            tooltip=(
                "The best few of every series stay; the rest go to the %s\n"
                "folder - NOT to rejected: they are fine frames, only surplus." % config.FOLDER_DUPLICATES
            ),
        )
        layout.addWidget(self.dedupe_button)
        self.clear_button = self._button(
            "Clear manual decisions", self._clear_manual_decisions,
            tooltip="Removes every manual override and duplicate mark;\nthe threshold decides again.",
        )
        layout.addWidget(self.clear_button)
        return section

    def _build_faces_section(self):
        """A separate pass: it needs the model file and the user decides if it is worth it."""
        section = CollapsibleSection("Faces", open_=False)
        layout = section.content_layout
        self.face_label = self._label()
        layout.addWidget(self.face_label)
        self.face_button = self._button(
            "Analyse faces", self._on_face_button,
            tooltip=(
                "Finds faces and measures the FACE sharpness separately, so a\n"
                "camera that focused on the background is caught. It also\n"
                "refines the best pick of each series. It never rejects."
            ),
        )
        layout.addWidget(self.face_button)
        return section

    def _build_enhance_section(self):
        """
        Correction strength. Closed at start (nothing to correct yet), it opens
        by itself when measuring ends - independent of whether anything was
        rejected (a hidden section once made correcting look tied to rejects).
        """
        section = CollapsibleSection("Correction", open_=False)
        layout = section.content_layout

        self.denoise_label, self.denoise_slider = self._strength(layout)
        self.sharpen_label, self.sharpen_slider = self._strength(layout)
        self.brightness_label, self.brightness_slider = self._strength(layout)
        self.brightness_slider.setToolTip(
            "Brightness relative to the MEASURED state:\n"
            "  dark frame   - lifted, but only to the bottom of the good band\n"
            "  good frame   - left alone\n"
            "  too bright   - pulled back, but only to the top of the band\n\n"
            "A gamma curve: highlights never clip."
        )
        self.wb_label, self.wb_slider = self._strength(layout, config.WB_SCALE_DEFAULT)
        self.wb_slider.setToolTip(WB_TOOLTIP)
        self.color_label, self.color_slider = self._strength(layout, config.COLOR_SCALE_DEFAULT)
        self.color_slider.setToolTip(
            "Saturation and local contrast relative to the MEASURED state:\n"
            "  dull frame       - saturation raised\n"
            "  hazy, flat frame - local contrast stretched\n"
            "  otherwise        - left alone\n\n"
            "Saturation cannot clip highlights; the contrast stretch holds\n"
            "itself back on noisy frames."
        )

        # the learned denoiser: far better, far slower, needs a separate download
        self.learned_check = QCheckBox("Learned denoiser (SCUNet)")
        self.learned_check.setCursor(Qt.PointingHandCursor)
        self.learned_check.stateChanged.connect(self._on_strength_changed)
        layout.addWidget(self.learned_check)
        self.learned_label = self._label()
        layout.addWidget(self.learned_label)

        self.strength_hint_label = self._label()
        layout.addWidget(self.strength_hint_label)
        self.preview_button = self._button(
            "Preview on one photo", self._open_preview,
            tooltip="Shows on a crop what the sliders do.\nAlso available with a right click on any card.",
        )
        layout.addWidget(self.preview_button)
        layout.addSpacing(4)
        # a separate button, not another main-button state: sorting MOVES,
        # correcting CREATES new files
        self.enhance_button = self._button("", self._on_enhance_button)
        layout.addWidget(self.enhance_button)
        return section

    def _strength(self, layout, start=None):
        label = self._label(wrap=False)
        layout.addWidget(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, config.ENHANCE_SCALE_MAX)
        slider.setValue(config.ENHANCE_SCALE_DEFAULT if start is None else start)
        slider.setSingleStep(25)  # 25 % steps, no pixel hunting
        slider.setPageStep(25)
        slider.valueChanged.connect(self._on_strength_changed)
        layout.addWidget(slider)
        return label, slider

    def _build_undo_button(self):
        """Not inside a section: it undoes both kinds of runs, and it is the
        button people look for in a panic."""
        self.undo_button = self._button("Undo last run", self._undo)
        self.undo_button.setFixedWidth(theme.PANEL_WIDTH)
        return self.undo_button

    def _build_grid(self):
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSpacing(8)
        self.grid.setUniformItemSizes(False)
        self.grid.setSelectionMode(QListWidget.NoSelection)
        self.grid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.empty_hint = QLabel(
            "Drop a folder or a single photo here, or use the buttons on the left."
        )
        self.empty_hint.setObjectName("dropHint")
        self.empty_hint.setAlignment(Qt.AlignCenter)
        self.empty_hint.setWordWrap(True)

        self.grid_stack = QStackedWidget()
        self.grid_stack.addWidget(self.empty_hint)
        self.grid_stack.addWidget(self.grid)
        return self.grid_stack

    def _build_footer(self):
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(4, 6, 4, 6)  # room for the signature glow
        layout.setSpacing(12)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(280)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = self._label(wrap=False)
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(SignatureButton())
        return footer

    def paintEvent(self, event):
        # stretched cached background: resizing does not re-run the blur
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(self.rect(), theme.background_pixmap())
        painter.end()

    # -----------------------------------------------------------------
    # folder and threshold
    # -----------------------------------------------------------------

    def busy(self):
        return self.state in BUSY_STATES

    def _choose_folder(self):
        # never while a run is active: its results belong to the current folder
        if self.busy():
            return
        chosen = QFileDialog.getExistingDirectory(self, "Choose a photo folder")
        if chosen:
            self.set_folder(Path(chosen))

    def set_folder(self, folder):
        if self.busy():
            return
        self.folder = Path(folder)
        self.folder_label.setText(elided(str(self.folder), self.folder_label, FOLDER_LABEL_WIDTH))
        self.folder_label.setToolTip(str(self.folder))
        self._clear_grid()
        self.results = []
        self.state = STATE_IDLE
        log.info("folder chosen: %s", self.folder)

        summary = shots_text(len(collect_images(self.folder)), "in the folder")
        last = last_run_summary(self.folder)
        if last and last["interrupted"]:
            summary += "\nThe last run in this folder was interrupted - Undo can restore it."
        self._set_summary(summary)
        self.empty_hint.setText("Press \"Start focus sorting\" to measure this folder.")
        self._refresh_undo()
        self._refresh_state()

    def _check_photo(self):
        if self.busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Check one photo", "", "Photos (*.jpg *.jpeg *.arw)")
        if path:
            self.open_quick_check(path)

    def open_quick_check(self, path):
        """The single-photo verdict (T-22): nothing is moved or logged."""
        if self.busy():
            return
        log.info("quick check: %s", path)
        dialog = QuickCheckDialog(
            path, self.threshold_slider.value(), self._strength_values(),
            self.learned_check.isChecked(), self,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.exec()

    def open_path(self, path):
        """A folder or a photo from the command line or a drop."""
        path = Path(path)
        if path.is_dir():
            self.set_folder(path)
        elif path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS:
            self.open_quick_check(path)

    def _dropped_path(self, event):
        mime = event.mimeData()
        urls = mime.urls() if mime.hasUrls() else []
        if len(urls) != 1 or not urls[0].isLocalFile() or self.busy():
            return None
        path = Path(urls[0].toLocalFile())
        if path.is_dir() or path.suffix.lower() in config.IMAGE_EXTENSIONS:
            return path
        return None

    def dragEnterEvent(self, event):
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._dropped_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        # not inside the drop handler: a modal dialog there would freeze Explorer
        QTimer.singleShot(0, lambda: self.open_path(path))

    def _on_threshold_changed(self, value):
        self.threshold_label.setText("Threshold: %d" % value)
        # live recolouring WITHOUT re-measuring
        for card in self.cards:
            card.set_threshold(value)
        if self.state == STATE_MEASURED:
            self._refresh_state()

    def _on_main_button(self):
        if self.state == STATE_IDLE:
            self._start_measuring()
        elif self.state == STATE_MEASURING:
            self._cancel(self.worker)
        elif self.state == STATE_MEASURED:
            self._move_files()

    def _cancel(self, worker):
        if worker is not None:
            worker.cancel()
            self.status_label.setText("Cancelling…")

    # -----------------------------------------------------------------
    # measuring
    # -----------------------------------------------------------------

    def _start_measuring(self):
        items = collect_images(self.folder)
        if not items:
            QMessageBox.information(self, "Keeper", "There are no photos to process in this folder.")
            return

        self._clear_grid()
        self.results = []
        self._show_progress(len(items))
        self.state = STATE_MEASURING
        self._refresh_state()
        self.status_panel.set_activity(ACTIVITY_MEASURING)
        self.status_panel.set_progress(0, len(items))
        log.info("measuring %d shots in %s", len(items), self.folder)

        self.worker = SharpnessWorker(items, self.folder, self)
        self.worker.progress.connect(self._on_measure_progress)
        self.worker.finished_all.connect(self._on_measure_finished)
        self.worker.finished.connect(self._worker_stopped)
        self.worker.start()

    def _from_stale_worker(self, current):
        """True for a signal of a worker that is no longer the current one."""
        sender = self.sender()
        return sender is not None and sender is not current

    def _on_measure_progress(self, done, total, result):
        if self._from_stale_worker(self.worker):
            return
        self.progress_bar.setValue(done)
        self.status_label.setText("Measuring: %d / %d" % (done, total))
        self.status_panel.set_activity(ACTIVITY_MEASURING, result.path.name)
        self.status_panel.set_progress(done, total)
        self.results.append(result)
        self._add_card(result)

    def _on_measure_finished(self, results, cancelled):
        if self._from_stale_worker(self.worker):
            return
        self.worker = None
        self.progress_bar.setVisible(False)
        self.status_panel.set_progress(None, None)
        self.status_panel.set_idle("Measuring cancelled" if cancelled else "Measuring finished")
        log.info("measuring %s: %d results", "cancelled" if cancelled else "finished", len(self.results))

        if cancelled:
            # partial results stay visible but cannot be moved: the picture is
            # incomplete (and series boundaries would be misleading)
            self.state = STATE_IDLE
            self.status_label.setText("Measuring cancelled.")
        else:
            self.state = STATE_MEASURED
            self.status_label.setText("Measuring finished.")
            self._detect_series()
            # correcting does not depend on rejects: open its section
            self.enhance_section.set_open(True)
        self._refresh_state()

    # -----------------------------------------------------------------
    # series
    # -----------------------------------------------------------------

    def _detect_series(self):
        """Only at the end: grouping needs every shot, partial groups would jump around."""
        self.series = series.group_results(self.results)
        mapping = series.info_map(self.series)
        for card in self.cards:
            card.set_series(mapping.get(card.result.path))
            card.set_faces(card.result.faces)
        self._refresh_cell_sizes()

    def _refresh_cell_sizes(self):
        """Series and face lines make cards taller; cells must follow or cards overlap."""
        for item, card in zip(self.grid_items, self.cards):
            item.setSizeHint(QSize(CARD_WIDTH, card.sizeHint().height()))

    def _recolor_cards(self):
        threshold = self.threshold_slider.value()
        for card in self.cards:
            card.set_threshold(threshold)
        self._refresh_state()

    def _protect_series_best(self):
        count = series.protect_best(self.series)
        self._recolor_cards()
        QMessageBox.information(
            self, "Keeper",
            "The best frame of %d series is protected.\n\nThese stay even if they score "
            "below the threshold." % count,
        )

    def _filter_duplicates(self):
        """Marks frames ABOVE the threshold too, so the numbers are shown first."""
        if not self.series:
            QMessageBox.information(self, "Keeper", "No series were found.")
            return
        keep = self.keep_spin.value()
        members = sum(s.size for s in self.series)
        kept = sum(min(keep, s.size) for s in self.series)
        answer = QMessageBox.question(
            self, "Keeper",
            "Of %d shots in %d series, %d stay and %d go to the %s folder.\n\n"
            "They are NOT rejects: fine frames that are only surplus, so they get "
            "their own folder.\n\nThis only marks them - nothing moves until you "
            "press the move button, and every card can still be overridden.\n\nContinue?"
            % (members, len(self.series), kept, members - kept, config.FOLDER_DUPLICATES),
        )
        if answer == QMessageBox.StandardButton.Yes:
            series.keep_best_n(self.series, keep)
            self._recolor_cards()

    def _clear_manual_decisions(self):
        count = clear_overrides(self.results) + series.clear_duplicates(self.results)
        self._recolor_cards()
        if count == 0:
            QMessageBox.information(self, "Keeper", "There were no manual decisions.")

    # -----------------------------------------------------------------
    # faces
    # -----------------------------------------------------------------

    def _on_face_button(self):
        if self.state == STATE_FACES:
            self._cancel(self.face_worker)
        else:
            self._start_faces()

    def _start_faces(self):
        readable = [r for r in self.results if r.score is not None]
        if not readable:
            return
        self._show_progress(len(readable))
        self.state = STATE_FACES
        self._refresh_state()
        self.status_panel.set_activity(ACTIVITY_FACES)
        self.status_panel.set_progress(0, len(readable))

        self.face_worker = FaceWorker(readable, self)
        self.face_worker.progress.connect(self._on_faces_progress)
        self.face_worker.finished_all.connect(self._on_faces_finished)
        self.face_worker.finished.connect(self._worker_stopped)
        self.face_worker.start()

    def _on_faces_progress(self, done, total, result):
        if self._from_stale_worker(self.face_worker):
            return
        self.progress_bar.setValue(done)
        self.status_panel.set_activity(ACTIVITY_FACES, result.path.name)
        self.status_panel.set_progress(done, total)
        self.status_label.setText(
            "Finding faces: %d / %d – %s (%d)" % (done, total, result.path.name, result.faces.count)
        )
        for card in self.cards:
            if card.result is result:
                card.set_faces(result.faces)
                break

    def _on_faces_finished(self, with_faces, cancelled):
        if self._from_stale_worker(self.face_worker):
            return
        self.face_worker = None
        self.progress_bar.setVisible(False)
        self.state = STATE_MEASURED
        self.status_panel.set_progress(None, None)
        self.status_panel.set_idle("Face analysis cancelled" if cancelled else "Face analysis finished")
        self.status_label.setText("Face analysis cancelled." if cancelled else "Face analysis finished.")
        # face sharpness changes the ranking inside series: pick the best again
        self._detect_series()
        self._realign_duplicates()
        self._refresh_state()

    def _realign_duplicates(self):
        """
        Existing duplicate marks follow the NEW ranking, otherwise a card could
        say "best" while being marked as a duplicate. Only if the user asked
        for duplicate marking before - nothing is marked unasked.
        """
        if any(r.duplicate for r in self.results):
            series.keep_best_n(self.series, self.keep_spin.value())
            self._recolor_cards()

    # -----------------------------------------------------------------
    # correction
    # -----------------------------------------------------------------

    def _kept_results(self, threshold):
        """The readable shots that stay (manual keeps included) - only these are corrected."""
        return [r for r in self.results if r.score is not None and is_kept(r, threshold)]

    def _sample_result(self):
        """The median-score kept shot: neither the best nor the worst tells about the batch."""
        kept = sorted(self._kept_results(self.threshold_slider.value()), key=lambda r: r.score)
        return kept[len(kept) // 2] if kept else None

    def _strength_values(self):
        return (
            self.denoise_slider.value(), self.sharpen_slider.value(), self.brightness_slider.value(),
            self.wb_slider.value(), self.color_slider.value(),
        )

    def _on_strength_changed(self, _value=None):
        """
        Says what the sliders MEAN on this batch: on a clean frame even 300 %
        does nothing (there is nothing to amplify) - better known in advance.
        """
        denoise, sharpen, brightness, wb, color = self._strength_values()
        self.denoise_label.setText("Denoising: %d%%" % denoise)
        self.sharpen_label.setText("Sharpening: %d%%" % sharpen)
        self.brightness_label.setText("Brightness: %d%%" % brightness)
        self.wb_label.setText("White balance: %d%%%s" % (wb, " (off)" if wb == 0 else ""))
        self.color_label.setText("Colour & local contrast: %d%%%s" % (color, " (off)" if color == 0 else ""))

        if not self.results:
            self.strength_hint_label.setText("After measuring, this explains what the sliders do to your photos.")
            return
        sample = self._sample_result()
        if sample is None:
            self.strength_hint_label.setText("No shot to correct.")
            return
        # noise is not re-measured on every slider move; sharpening depends
        # only on the score, so it can be stated exactly
        _, amount = scaled_strengths(None, sample.score, denoise, sharpen)
        if amount <= 0.0:
            text = "The typical shot (%d points) gets no sharpening: that score is past saving." % round(sample.score)
        else:
            text = (
                "The typical shot (%d points) gets sharpening strength %.2f. "
                "Denoising follows the noise measured during the run." % (round(sample.score), amount)
            )
        self.strength_hint_label.setText(text)

    def _open_preview(self, result=None):
        """Correction preview: a card's own shot, or the batch's typical shot."""
        # the button's clicked signal passes a bool, not a result
        if not hasattr(result, "path"):
            result = self._sample_result()
        if result is None:
            QMessageBox.information(self, "Keeper", "Measure first, then there is something to preview.")
            return

        denoise, sharpen, brightness, wb, color = self._strength_values()
        dialog = PreviewDialog(result, denoise, sharpen, brightness, wb,
                               self.learned_check.isChecked(), color, self)
        try:
            # only an explicit "Use these settings" changes the sliders (Esc does not)
            if dialog.exec() == QDialog.Accepted:
                for slider, value in zip(
                    (self.denoise_slider, self.sharpen_slider, self.brightness_slider,
                     self.wb_slider, self.color_slider),
                    dialog.values(),
                ):
                    slider.setValue(value)
        finally:
            dialog.deleteLater()   # no hidden dialogs accumulate

    def _on_enhance_button(self):
        if self.state == STATE_ENHANCING:
            self._cancel(self.enhance_worker)
        else:
            self._start_enhancing()

    def _start_enhancing(self):
        kept = self._kept_results(self.threshold_slider.value())
        if not kept:
            QMessageBox.information(self, "Keeper", "No shot stays at the current threshold.")
            return
        # correcting is slow and says so up front, or it looks frozen
        answer = QMessageBox.question(
            self, "Keeper",
            "Corrected copies of %s will be saved to the %s folder.\n\n"
            "The originals are not touched.\n\nThis takes a few seconds per photo.\n\nContinue?"
            % (shots_text(len(kept), "").strip(), config.FOLDER_CORRECTED),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._show_progress(len(kept))
        self.state = STATE_ENHANCING
        self._refresh_state()
        self.status_panel.set_activity(ACTIVITY_ENHANCING)
        self.status_panel.set_progress(0, len(kept))
        denoise, sharpen, brightness, wb, color = self._strength_values()
        log.info("correcting %d shots in %s", len(kept), self.folder)

        self.enhance_worker = EnhanceWorker(
            self.folder, kept, denoise, sharpen, brightness, wb,
            self.learned_check.isChecked(), color, self,
        )
        self.enhance_worker.progress.connect(self._on_enhance_progress)
        self.enhance_worker.finished_all.connect(self._on_enhance_finished)
        self.enhance_worker.finished.connect(self._worker_stopped)
        self.enhance_worker.start()

    def _on_enhance_progress(self, done, total, progress):
        if self._from_stale_worker(self.enhance_worker):
            return
        self.progress_bar.setValue(done)
        self.status_panel.set_activity(ACTIVITY_ENHANCING, progress.path.name)
        self.status_panel.set_progress(done, total)
        # WHAT happened, so it is visible that each frame gets its own treatment
        if progress.error is not None:
            detail = "error"
        elif not progress.anything_changed:
            detail = "nothing to correct"
        else:
            denoiser = "learned" if progress.learned else "%.0f" % progress.denoise_h
            detail = "noise %.1f · filter %s · sharpening %.2f" % (
                progress.sigma or 0.0, denoiser, progress.sharpen_amount
            )
            if progress.tone_changed:
                detail += " · light/colour adjusted"
        self.status_label.setText("Correcting: %d / %d – %s (%s)" % (done, total, progress.path.name, detail))

    def _on_enhance_finished(self, report, cancelled):
        if self._from_stale_worker(self.enhance_worker):
            return
        self.enhance_worker = None
        self.progress_bar.setVisible(False)
        self.state = STATE_MEASURED
        self.status_panel.set_progress(None, None)
        self.status_panel.set_idle("Correction cancelled" if cancelled else "Correction finished")
        self._refresh_undo()
        log.info("correction: %d saved, %d failed, error=%s", len(report.saved), len(report.failed), report.error)

        if report.error is not None:
            QMessageBox.warning(self, "Keeper", "The correction stopped.\n\n%s" % report.error)
            self.status_label.setText("The correction stopped with an error.")
            self._refresh_state()
            return

        message = "%d corrected cop%s saved to the %s folder." % (
            len(report.saved), "y" if len(report.saved) == 1 else "ies", config.FOLDER_CORRECTED
        )
        if cancelled:
            message = "The correction was cancelled.\n\n" + message
        if report.failed:
            message += "\n\n%d shot%s skipped:\n" % (len(report.failed), "" if len(report.failed) == 1 else "s")
            message += "\n".join("%s – %s" % (Path(p).name, reason) for p, reason in report.failed[:5])
        QMessageBox.information(self, "Keeper", message)
        self.status_label.setText("Correction cancelled." if cancelled else "Correction finished.")
        self._refresh_state()

    # -----------------------------------------------------------------
    # moving and undo
    # -----------------------------------------------------------------

    def _move_files(self):
        threshold = self.threshold_slider.value()
        counts = self._count(threshold)
        shots = counts["rejects"] + counts["duplicates"]
        if shots == 0:
            QMessageBox.information(
                self, "Keeper", "No shot falls below the threshold and no duplicate is marked."
            )
            return

        # a RAW+JPEG pair is two files: say so if the numbers differ
        files = sum(len(r.files) for r in self.results if destination_folder(r, threshold) is not None)
        amount = shots_text(shots, "").strip() + ("" if files == shots else " (%d files)" % files)
        # name both folders: rejects are faults, duplicates are fine frames
        where = []
        if counts["rejects"]:
            where.append("%d to the %s folder" % (counts["rejects"], config.FOLDER_REJECTS))
        if counts["duplicates"]:
            where.append("%d to the %s folder" % (counts["duplicates"], config.FOLDER_DUPLICATES))
        answer = QMessageBox.question(
            self, "Keeper",
            "%s will move: %s.\n\nThis can be undone at any time.\n\nContinue?" % (amount, " and ".join(where)),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        report = move_to_rejects(self.folder, self.results, threshold)
        log.info("moved %d files, %d failed, error=%s", len(report.moved), len(report.failed), report.error)
        self._refresh_undo()

        # read-only folder or damaged log: nothing moved
        if report.error is not None:
            QMessageBox.warning(self, "Keeper", "Nothing was moved.\n\n%s" % report.error)
            self._refresh_state()
            return

        message = "%d file%s moved." % (len(report.moved), "" if len(report.moved) == 1 else "s")
        if report.failed:
            message += "\n\n%d could not be moved:\n" % len(report.failed)
            message += "\n".join("%s – %s" % (Path(p).name, reason) for p, reason in report.failed[:5])
        QMessageBox.information(self, "Keeper", message)

        # the survivors stay on the grid (their scores are still valid); what
        # survived is asked from the DISK, a failed move stays where it was
        self.results = [r for r in self.results if all(f.exists() for f in r.files)]
        self._clear_grid()
        for result in self.results:
            self._add_card(result)
        if self.results:
            self.state = STATE_MEASURED
            self._detect_series()
        else:
            self.state = STATE_IDLE
        self._set_summary(shots_text(len(collect_images(self.folder)), "left in the folder"))
        self.status_label.setText("Moving finished.")
        self._refresh_state()

    def _undo(self):
        """Undoes the LAST run, whichever kind - after a confirmation (S-3)."""
        if self.busy() or self.folder is None:
            return
        summary = last_run_summary(self.folder)
        if summary is None:
            self._refresh_undo()
            self._refresh_state()
            return

        correction = summary["kind"] == KIND_CORRECTION
        what = (
            "remove %d corrected cop%s (copies you edited since are kept)"
            % (summary["count"], "y" if summary["count"] == 1 else "ies")
            if correction else "move %d file%s back" % (summary["count"], "" if summary["count"] == 1 else "s")
        )
        answer = QMessageBox.question(
            self, "Keeper",
            "Undo the last run in\n%s?\n\nThis will %s.\nRun: %s%s"
            % (self.folder, what, summary["timestamp"] or "unknown time",
               " (interrupted)" if summary["interrupted"] else ""),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            report = undo_last_correction(self.folder) if correction else undo_last_run(self.folder)
        except Exception as exc:
            # the log can change between the check and the click
            log.exception("undo failed")
            QMessageBox.warning(self, "Keeper", "The undo failed.\n\n%s" % exc)
            self._refresh_undo()
            self._refresh_state()
            return
        log.info("undo: %d restored, %d skipped", len(report.restored), len(report.skipped))

        if correction:
            message = "%d corrected cop%s removed." % (len(report.restored), "y" if len(report.restored) == 1 else "ies")
        else:
            message = "%d file%s restored." % (len(report.restored), "" if len(report.restored) == 1 else "s")
        if report.skipped:
            message += "\n\n%d skipped:\n" % len(report.skipped)
            message += "\n".join("%s – %s" % (Path(str(p)).name, reason) for p, reason in report.skipped[:5])
        if report.remaining:
            message += "\n\n%d entr%s kept for another try." % (report.remaining, "y" if report.remaining == 1 else "ies")
        QMessageBox.information(self, "Keeper", message)

        self._clear_grid()
        self.results = []
        self.state = STATE_IDLE
        self._set_summary(shots_text(len(collect_images(self.folder)), "in the folder"))
        self.status_label.setText("Undo finished.")
        self.status_panel.set_idle("Undo finished")
        self._refresh_undo()
        self._refresh_state()

    # -----------------------------------------------------------------
    # grid and state
    # -----------------------------------------------------------------

    def _show_progress(self, total):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)

    def _clear_grid(self):
        self.grid.clear()
        self.cards = []
        self.grid_items = []
        self.series = []
        self.grid_stack.setCurrentWidget(self.empty_hint)

    def _add_card(self, result):
        card = ResultCard(result, self.threshold_slider.value())
        card.override_changed.connect(self._refresh_state)
        card.preview_requested.connect(self._open_preview)
        item = QListWidgetItem(self.grid)
        # the KNOWN card width, not the sizeHint (a long label would make the
        # cell narrower than the card and neighbours would overlap)
        item.setSizeHint(QSize(CARD_WIDTH, card.sizeHint().height()))
        self.grid.setItemWidget(item, card)
        self.cards.append(card)
        self.grid_items.append(item)
        self.grid_stack.setCurrentWidget(self.grid)

    def _count(self, threshold):
        """Counts with the same destination_folder() the actual move uses."""
        counts = {"kept": 0, "rejects": 0, "duplicates": 0, "unreadable": 0, "manual": 0}
        for result in self.results:
            if result.score is None:
                counts["unreadable"] += 1
                continue
            if result.override is not None:
                counts["manual"] += 1
            target = destination_folder(result, threshold)
            if target == config.FOLDER_REJECTS:
                counts["rejects"] += 1
            elif target == config.FOLDER_DUPLICATES:
                counts["duplicates"] += 1
            else:
                counts["kept"] += 1
        return counts

    def _set_summary(self, text):
        self.summary_label.setText(text)
        self.summary_label.setVisible(bool(text))

    def _main_button_text(self, counts):
        rejects, duplicates = counts["rejects"], counts["duplicates"]
        if rejects and duplicates:
            return "Move %d reject%s + %d duplicate%s" % (
                rejects, "" if rejects == 1 else "s", duplicates, "" if duplicates == 1 else "s"
            )
        if rejects:
            return "Move %d reject%s" % (rejects, "" if rejects == 1 else "s")
        if duplicates:
            return "Move %d duplicate%s" % (duplicates, "" if duplicates == 1 else "s")
        # say it plainly; a greyed "(0)" read like the program got stuck
        return "Nothing to move"

    def _refresh_state(self):
        """The ONE place that enables or disables controls for the current state."""
        busy = self.busy()
        measured = self.state == STATE_MEASURED
        threshold = self.threshold_slider.value()
        self.threshold_label.setText("Threshold: %d" % threshold)
        self.folder_button.setEnabled(not busy)
        self.photo_button.setEnabled(not busy)

        if self.state == STATE_MEASURING:
            self.main_button.setText("Cancel")
            self.main_button.setEnabled(True)
        elif measured:
            counts = self._count(threshold)
            self.main_button.setText(self._main_button_text(counts))
            self.main_button.setEnabled(counts["rejects"] + counts["duplicates"] > 0)
            summary = "%s\nKept: %d · Rejects: %d · Duplicates: %d · Unreadable: %d" % (
                shots_text(len(self.results), "in total"), counts["kept"], counts["rejects"],
                counts["duplicates"], counts["unreadable"],
            )
            if counts["manual"]:
                summary += "\nOverridden by hand: %d" % counts["manual"]
            if not counts["rejects"] and not counts["duplicates"] and counts["kept"]:
                summary += "\n\nEvery shot passed. Correcting is still possible – see Correction."
            self._set_summary(summary)
        elif busy:
            self.main_button.setText("Start focus sorting")
            self.main_button.setEnabled(False)
        else:
            self.main_button.setText("Start focus sorting")
            self.main_button.setEnabled(self.folder is not None)

        # the preview needs something to show and free cores
        self.preview_button.setEnabled(measured and self._sample_result() is not None)
        if self.state == STATE_ENHANCING:
            self.enhance_button.setText("Cancel correction")
            self.enhance_button.setEnabled(True)
        elif measured:
            kept = len(self._kept_results(threshold))
            self.enhance_button.setText("Save corrected copies (%d)" % kept)
            self.enhance_button.setEnabled(kept > 0)
        else:
            self.enhance_button.setText("Save corrected copies")
            self.enhance_button.setEnabled(False)

        self._refresh_series_buttons()
        self._refresh_face_button()
        self._refresh_learned_toggle()
        self.undo_button.setEnabled(self.folder is not None and not busy and self._undo_available)
        self._on_strength_changed()

    def _refresh_undo(self):
        self._undo_available = self.folder is not None and has_undoable_run(self.folder)

    def _refresh_learned_toggle(self):
        """A disabled checkbox says WHY (missing PyTorch, GPU or weights)."""
        missing = learned_denoise.unavailable_reason()
        if missing is not None:
            self.learned_check.setChecked(False)
            self.learned_check.setEnabled(False)
            self.learned_label.setText(missing)
            return
        self.learned_check.setEnabled(True)
        if self.learned_check.isChecked():
            self.learned_label.setText(
                "A neural network replaces the classic filter. Much cleaner, but about "
                "a minute per photo – hours for a large batch."
            )
        else:
            self.learned_label.setText(
                "When on: far cleaner results (36.6 dB instead of 20.5 dB measured), "
                "at about a minute per photo instead of ten seconds."
            )

    def _refresh_face_button(self):
        """A missing model is explained, not hidden behind a dead button."""
        if not faces.model_available():
            self.face_label.setText(
                "Needs the ~230 kB face model, which is missing from assets/. "
                "The README section \"Faces\" says where to get it."
            )
            self.face_button.setText("Analyse faces")
            self.face_button.setEnabled(False)
            return
        if self.state == STATE_FACES:
            self.face_button.setText("Cancel face analysis")
            self.face_button.setEnabled(True)
            return

        analysed = [r for r in self.results if r.faces is not None]
        if analysed:
            with_faces = sum(1 for r in analysed if r.faces.count)
            soft = sum(
                1 for r in analysed
                if r.faces.main is not None and r.faces.main.sharpness is not None
                and r.faces.main.sharpness < config.FACE_SOFT_THRESHOLD
            )
            text = "%s with faces." % shots_text(with_faces, "").strip()
            if soft:
                text += " %d with a soft main face." % soft
            self.face_label.setText(text)
        else:
            self.face_label.setText(
                "Measures face sharpness separately, so a camera that focused on the "
                "background is caught. Also refines the best pick of each series."
            )
        self.face_button.setText("Analyse faces")
        self.face_button.setEnabled(self.state == STATE_MEASURED)

    def _refresh_series_buttons(self):
        measured = self.state == STATE_MEASURED
        if not measured:
            self.series_label.setText("After measuring, this shows how many series were found.")
        elif not self.series:
            self.series_label.setText("No series found: every shot is its own moment.")
        else:
            members = sum(s.size for s in self.series)
            self.series_label.setText(
                "%d series · %s · longest %d" % (
                    len(self.series), shots_text(members, "").strip(), max(s.size for s in self.series)
                )
            )
        self.protect_button.setEnabled(measured and bool(self.series))
        self.dedupe_button.setEnabled(measured and bool(self.series))
        self.keep_spin.setEnabled(measured and bool(self.series))
        # cards can be overridden one by one too, independent of series
        self.clear_button.setEnabled(
            measured and any(r.override is not None or r.duplicate for r in self.results)
        )

    def changeEvent(self, event):
        # the system can maximise too (Win+Up), so the glyph follows the state
        if event.type() == QEvent.WindowStateChange:
            self.title_bar.update_glyph()
        super().changeEvent(event)

    # -----------------------------------------------------------------
    # closing
    # -----------------------------------------------------------------

    def _workers(self):
        return [w for w in (self.worker, self.enhance_worker, self.face_worker) if w is not None]

    def _worker_stopped(self):
        # a close was requested while a run was active: finish it now
        if self._closing and not any(w.isRunning() for w in self._workers()):
            self.close()

    def closeEvent(self, event):
        """
        Never blocks: a running worker is cancelled, the window disappears at
        once, and the real close happens when the worker has stopped (an image
        step cannot be interrupted midway; correcting stops between tiles).
        """
        running = [w for w in self._workers() if w.isRunning()]
        if running:
            self._closing = True
            for worker in running:
                worker.cancel()
            self.hide()
            event.ignore()
            return
        self.status_panel.stop_monitoring()
        super().closeEvent(event)
