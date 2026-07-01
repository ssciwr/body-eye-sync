from __future__ import annotations

import threading
from importlib.resources import as_file, files
from pathlib import Path

from qtpy.QtCore import Slot
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

from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.face_detection_worker import FaceDetectionWorker
from body_eye_sync.gui.object_tracking_worker import ObjectTrackingWorker
from body_eye_sync.gui.video_viewer import VideoViewer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("body-eye-sync")
        with as_file(files(__package__) / "resources" / "icon.ico") as icon_path:
            self.setWindowIcon(QIcon(str(icon_path)))

        self.video = Video()
        self._thread: threading.Thread | None = None
        self._worker: ObjectTrackingWorker | FaceDetectionWorker | None = None
        self._in_setup = False

        self.open_button = QPushButton("Open video…")
        self.open_button.clicked.connect(self._choose_video)
        self.file_label = QLabel("No file selected")

        self.run_button = QPushButton("Run object tracking")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._start_object_tracking)

        self.face_button = QPushButton("Run face detection")
        self.face_button.setEnabled(False)
        self.face_button.clicked.connect(self._start_face_detection)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_run)

        self.video_viewer = VideoViewer()

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.open_button)
        top_bar.addWidget(self.file_label, stretch=1)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.run_button)
        bottom_bar.addWidget(self.face_button)
        bottom_bar.addWidget(self.progress_bar, stretch=1)
        bottom_bar.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.video_viewer, stretch=1)
        layout.addLayout(bottom_bar)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

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
        self.video.set_video(path)
        try:
            self.video_viewer.load(self.video)
        except OSError as exc:
            QMessageBox.critical(self, "Could not open video", str(exc))
            return
        self.file_label.setText(str(path))
        self.run_button.setEnabled(True)
        # A fresh video has no object tracking results yet, so face detection waits.
        self.face_button.setEnabled(False)

    def _start_object_tracking(self) -> None:
        if self.video.video_path is None or self._thread is not None:
            return

        self.video.begin_object_tracking()
        self._begin_run()

        self._worker = ObjectTrackingWorker(self.video)
        self._worker.new_frame.connect(self._on_new_frame)
        self._worker.new_frame.connect(self.video_viewer.show_live_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _start_face_detection(self) -> None:
        if self.video.data is None or self._thread is not None:
            return

        # Keep the tracked boxes; only the previous face columns are discarded.
        self.video.begin_face_detection()
        self._begin_run()

        self._worker = FaceDetectionWorker(self.video)
        self._worker.new_frame.connect(self._on_new_frame)
        self._worker.new_frame.connect(self.video_viewer.show_live_face_frame)
        self._worker.finished.connect(self._on_face_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _begin_run(self) -> None:
        """Shared start-up for object tracking and face detection runs."""
        self._set_running(True)
        # Weights are built/downloaded before the first frame is processed, so
        # show a busy bar until the first frame arrives.
        self._in_setup = True
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Downloading weights…")

    @Slot(object)
    def _on_new_frame(self, frame) -> None:
        total = self.video_viewer.frame_count
        if self._in_setup:
            # First frame processed: switch from the busy "downloading" bar to a
            # determinate progress bar (a 0..0 range stays busy if total unknown).
            self._in_setup = False
            self.progress_bar.setRange(0, total)
            self.progress_bar.setFormat("%p%" if total else "Object tracking…")
        if total:
            self.progress_bar.setValue(frame.frame_idx)

    def _cancel_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")

    @Slot()
    def _on_finished(self) -> None:
        data = self.video.data
        self.statusBar().showMessage(
            f"Object tracking finished: {data['track_id'].nunique()} tracklets, "
            f"{len(data)} detections"
        )
        self._set_running(False)

    @Slot()
    def _on_face_finished(self) -> None:
        data = self.video.data
        n_faces = int(data["face_score"].notna().sum())
        self.statusBar().showMessage(
            f"Face detection finished: {n_faces} faces over {len(data)} detections"
        )
        self._set_running(False)

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(f"{self._worker.operation_name} failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self.statusBar().showMessage(f"{self._worker.operation_name} cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            # background thread has reported back; drop our references to it.
            self._thread = None
            self._worker = None
            self.video_viewer.refresh_overlays()
        self.run_button.setEnabled(not running and self.video.video_path is not None)
        # Face detection runs on the tracked boxes, so it needs object tracking results.
        self.face_button.setEnabled(not running and self.video.data is not None)
        self.open_button.setEnabled(not running)
        self.video_viewer.enable_controls(not running)
        self.progress_bar.setVisible(running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        super().closeEvent(event)
