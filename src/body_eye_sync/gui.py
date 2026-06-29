from __future__ import annotations

import threading
from pathlib import Path

from qtpy.QtCore import QObject, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.state import AppState
from body_eye_sync.video_viewer import VideoViewer

_ICON = Path(__file__).parent / "resources" / "icon.ico"


class TrackingWorker(QObject):
    """Runs :func:`detect_tracklets` off the GUI thread."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, video_path: str | Path) -> None:
        super().__init__()
        self._video_path = video_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        try:
            # Imported here, not at module load, so the heavy boxmot/torch stack
            # only loads when tracking is actually run (keeps GUI startup fast).
            from body_eye_sync.tracking import detect_tracklets

            tracklets = detect_tracklets(
                self._video_path,
                progress=self.progress.emit,
                is_cancelled=self._cancel.is_set,
            )
        except Exception as exc:  # surface any failure to the GUI
            self.failed.emit(str(exc))
            return
        if self._cancel.is_set():
            self.cancelled.emit()
        else:
            self.finished.emit(tracklets)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("body-eye-sync")
        self.setWindowIcon(QIcon(str(_ICON)))

        self.state = AppState()
        self._thread: threading.Thread | None = None
        self._worker: TrackingWorker | None = None
        self._in_setup = False

        self.open_button = QPushButton("Open video…")
        self.open_button.clicked.connect(self._choose_video)
        self.file_label = QLabel("No file selected")

        self.run_button = QPushButton("Run tracking")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._start_tracking)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_tracking)

        self.video_viewer = VideoViewer()
        self.video_viewer.set_box_provider(self.state.boxes_for_frame)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.open_button)
        top_bar.addWidget(self.file_label, stretch=1)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.run_button)
        bottom_bar.addWidget(self.progress_bar, stretch=1)
        bottom_bar.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.video_viewer, stretch=1)
        layout.addLayout(bottom_bar)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self._load_video(Path(path))

    def _load_video(self, path: Path) -> None:
        try:
            self.video_viewer.load(path)
        except OSError as exc:
            QMessageBox.critical(self, "Could not open video", str(exc))
            return
        # Only commit the new video to state once it has opened successfully.
        # set_video() clears any previous tracklets, so refresh the overlays to
        # drop boxes from the old video off the freshly displayed first frame.
        self.state.set_video(path)
        self.video_viewer.refresh_overlays()
        self.file_label.setText(str(path))
        self.run_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    def _start_tracking(self) -> None:
        if self.state.video_path is None or self._thread is not None:
            return

        self._set_running(True)
        # Weights are built/downloaded before the first frame is processed, so
        # show a busy bar until the first progress callback arrives.
        self._in_setup = True
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Downloading weights…")

        # The worker lives in the GUI thread, so its signals are auto-queued to
        # this thread when emitted from the background thread.
        self._worker = TrackingWorker(self.state.video_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    @Slot(int)
    def _on_progress(self, frame_idx: int) -> None:
        total = self.video_viewer.frame_count
        if self._in_setup:
            # First frame processed: switch from the busy "downloading" bar to a
            # determinate progress bar (a 0..0 range stays busy if total unknown).
            self._in_setup = False
            self.progress_bar.setRange(0, total)
            self.progress_bar.setFormat("%p%" if total else "Tracking…")
        if total:
            self.progress_bar.setValue(frame_idx)

    def _cancel_tracking(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")

    @Slot(object)
    def _on_finished(self, tracklets) -> None:
        self.state.set_tracklets(tracklets)
        self.video_viewer.refresh_overlays()
        self.statusBar().showMessage(
            f"Tracking finished: {tracklets['track_id'].nunique()} tracklets, "
            f"{len(tracklets)} detections"
        )
        self._set_running(False)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Tracking failed", message)
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self.statusBar().showMessage("Tracking cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            # The background thread has reported back; drop our references to it.
            self._thread = None
            self._worker = None
        self.run_button.setEnabled(not running and self.state.video_path is not None)
        self.open_button.setEnabled(not running)
        self.progress_bar.setVisible(running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")

    def closeEvent(self, event) -> None:
        # Make sure a running tracking thread stops cleanly before we exit.
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        super().closeEvent(event)
