#!/usr/bin/env python
"""Tabbed desktop GUI for running and inspecting the AMGAZE pipeline."""

from __future__ import annotations

import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
from qtpy.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from qtpy.QtGui import QImage, QPixmap, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .pipeline_backend import (
    AlignmentConfig,
    ClusteringConfig,
    FaceConfig,
    PipelinePaths,
    TrackingConfig,
    draw_preview_frame,
    parse_gaze,
    parse_gaze_on_faces,
    parse_landmarks,
    parse_tracks,
    run_alignment,
    run_clustering,
    run_face_preprocessing,
    run_tracking,
)


class SignalStream(QObject):
    text_written = Signal(str)

    def write(self, text):
        if text:
            self.text_written.emit(str(text))
        return len(text)

    def flush(self):
        pass


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, function, stream):
        super().__init__()
        self.function = function
        self.stream = stream

    @Slot()
    def run(self):
        try:
            with redirect_stdout(self.stream), redirect_stderr(self.stream):
                result = self.function()
        except BaseException:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


class VideoPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.video = QLabel("Run this stage to load its preview")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        self.video.setStyleSheet("background: #181818; color: #aaa")
        self.play = QPushButton("Play")
        self.play.clicked.connect(self.toggle_playback)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        self.position = QLabel("0 / 0")

        controls = QHBoxLayout()
        controls.addWidget(self.play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.position)
        layout = QVBoxLayout(self)
        layout.addWidget(self.video, 1)
        layout.addLayout(controls)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.cap = None
        self.frame_count = 0
        self.fps = 25.0
        self.current_frame = 0
        self.frame_offset = -1
        self.tracks = {}
        self.landmarks = {}
        self.gaze = None
        self.gaze_on_faces = {}

    def close_video(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.play.setText("Play")

    def reset(self, message="Run this stage to load its preview"):
        self.close_video()
        self.video.clear()
        self.video.setText(message)
        self.slider.setRange(0, 0)
        self.position.setText("0 / 0")
        self.tracks = {}
        self.landmarks = {}
        self.gaze = None
        self.gaze_on_faces = {}

    def load(
        self,
        video_path: Path,
        tracks_path: Path | None = None,
        person_labels: bool = False,
        landmarks_path: Path | None = None,
        gaze_path: Path | None = None,
        gaze_on_faces_path: Path | None = None,
        frame_offset: int = -1,
    ):
        self.close_video()
        if not video_path.is_file():
            self.video.setText(f"Video not found: {video_path}")
            return
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            self.video.setText(f"Could not open: {video_path}")
            self.cap = None
            return
        self.frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_offset = frame_offset
        self.tracks = parse_tracks(tracks_path, person_labels)
        self.landmarks = parse_landmarks(landmarks_path)
        self.gaze = parse_gaze(gaze_path)
        self.gaze_on_faces = parse_gaze_on_faces(gaze_on_faces_path)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.frame_count - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.current_frame = 0
        self.timer.setInterval(max(1, round(1000 / self.fps)))
        self.show_frame(0)

    def show_frame(self, index):
        if self.cap is None:
            return
        index = max(0, min(self.frame_count - 1, int(index)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.play.setText("Play")
            return
        frame = draw_preview_frame(
            frame,
            index,
            self.fps,
            self.frame_offset,
            self.tracks,
            self.landmarks,
            self.gaze,
            self.gaze_on_faces,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video.setPixmap(pixmap)
        self.current_frame = index
        self.position.setText(f"{index + 1} / {self.frame_count}")
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

    def toggle_playback(self):
        if self.cap is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Play")
        else:
            if self.current_frame >= self.frame_count - 1:
                self.show_frame(0)
            self.timer.start()
            self.play.setText("Pause")

    def next_frame(self):
        if self.current_frame >= self.frame_count - 1:
            self.timer.stop()
            self.play.setText("Play")
        else:
            self.show_frame(self.current_frame + 1)

    def seek(self, value):
        self.show_frame(value)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.cap is not None:
            self.show_frame(self.current_frame)


class PathField(QWidget):
    def __init__(self, value="", directory=False, file_filter="All files (*)"):
        super().__init__()
        self.directory = directory
        self.file_filter = file_filter
        self.edit = QLineEdit(str(value))
        button = QPushButton("Browse…")
        button.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def browse(self):
        if self.directory:
            value = QFileDialog.getExistingDirectory(
                self, "Select directory", self.edit.text()
            )
        else:
            value, _ = QFileDialog.getOpenFileName(
                self, "Select file", self.edit.text(), self.file_filter
            )
        if value:
            self.edit.setText(value)

    def path(self):
        return Path(self.edit.text()).expanduser()


class StageTab(QWidget):
    def __init__(self, run_label):
        super().__init__()
        self.form = QFormLayout()
        self.run_button = QPushButton(run_label)
        self.status = QLabel("Ready")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.preview = VideoPreview()

        left = QVBoxLayout()
        left.addLayout(self.form)
        left.addWidget(self.run_button)
        left.addWidget(self.status)
        left.addWidget(QLabel("Stage log"))
        left.addWidget(self.log, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(390)
        left_widget.setMaximumWidth(520)

        layout = QHBoxLayout(self)
        layout.addWidget(left_widget)
        layout.addWidget(self.preview, 1)

    def append_log(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMGAZE pipeline")
        self.resize(1320, 800)
        self.thread = None
        self.worker = None
        self.stream = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tracking_tab = self.make_tracking_tab()
        self.alignment_tab = self.make_alignment_tab()
        self.face_tab = self.make_face_tab()
        self.clustering_tab = self.make_clustering_tab()
        self.tabs.addTab(self.tracking_tab, "1. Tracking")
        self.tabs.addTab(self.alignment_tab, "2. Gaze alignment")
        self.tabs.addTab(self.face_tab, "3. Faces / ReID")
        self.tabs.addTab(self.clustering_tab, "4. Clustering")
        self.refresh_available_previews()

    def make_tracking_tab(self):
        tab = StageTab("Run BoxMOT tracking")
        self.video_field = PathField(
            "Archiv/quartet.mp4",
            file_filter="Videos (*.mp4 *.mov *.avi);;All files (*)",
        )
        self.gaze_field = PathField(
            "Archiv/G3_1401mp4.tsv", file_filter="TSV files (*.tsv);;All files (*)"
        )
        self.workspace_field = PathField("runs/track/quartet2", directory=True)
        self.detector = QLineEdit("yolo26l")
        self.boxmot_reid = QLineEdit("osnet_x0_25_msmt17")
        self.tracker = QLineEdit("botsort")
        self.device = QLineEdit("0")
        tab.form.addRow("Video", self.video_field)
        tab.form.addRow("Gaze TSV", self.gaze_field)
        tab.form.addRow("Run directory", self.workspace_field)
        tab.form.addRow("Detector", self.detector)
        tab.form.addRow("BoxMOT ReID", self.boxmot_reid)
        tab.form.addRow("Tracker", self.tracker)
        tab.form.addRow("Device", self.device)
        self.load_workspace_button = QPushButton("Load existing workspace")
        self.load_workspace_button.clicked.connect(
            lambda: self.load_existing_workspace(show_message=True)
        )
        tab.form.addRow("", self.load_workspace_button)
        tab.run_button.clicked.connect(self.start_tracking)
        return tab

    def make_alignment_tab(self):
        tab = StageTab("Align gaze timestamps")
        self.time_col = QLineEdit("gaze_video_time")
        self.reuse_markers = QCheckBox("Reuse detected video markers when available")
        self.reuse_markers.setChecked(True)
        tab.form.addRow("TSV time column", self.time_col)
        tab.form.addRow("", self.reuse_markers)
        tab.run_button.clicked.connect(self.start_alignment)
        return tab

    def make_face_tab(self):
        tab = StageTab("Extract faces, landmarks, and embeddings")
        self.face_model = QLineEdit("buffalo_l")
        self.face_threshold = QDoubleSpinBox()
        self.face_threshold.setRange(0.0, 1.0)
        self.face_threshold.setSingleStep(0.05)
        self.face_threshold.setValue(0.55)
        self.max_samples = QSpinBox()
        self.max_samples.setRange(1, 1000)
        self.max_samples.setValue(30)
        self.landmark_stride = QSpinBox()
        self.landmark_stride.setRange(1, 1000)
        self.landmark_stride.setValue(1)
        self.reid_weights = PathField(
            "weights/osnet_x1_0_msmt17.pt",
            file_filter="Model files (*.pt);;All files (*)",
        )
        tab.form.addRow("Face model", self.face_model)
        tab.form.addRow("Face threshold", self.face_threshold)
        tab.form.addRow("Samples per tracklet", self.max_samples)
        tab.form.addRow("Landmark frame stride", self.landmark_stride)
        tab.form.addRow("Body ReID weights", self.reid_weights)
        tab.run_button.clicked.connect(self.start_faces)
        return tab

    def make_clustering_tab(self):
        tab = StageTab("Run cached clustering")
        constraint_note = QLabel(
            "Co-present tracklets always receive different person IDs."
        )
        constraint_note.setWordWrap(True)
        self.fixed_layout = QCheckBox("Use fixed left/right layout constraints")
        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(0, 1000)
        self.cluster_count.setSpecialValueText("Automatic")
        self.distance_threshold = QDoubleSpinBox()
        self.distance_threshold.setRange(0.0, 2.0)
        self.distance_threshold.setDecimals(3)
        self.distance_threshold.setSingleStep(0.05)
        self.distance_threshold.setSpecialValueText("Automatic")
        self.kmax = QSpinBox()
        self.kmax.setRange(2, 1000)
        self.kmax.setValue(10)
        self.layout_weight = QDoubleSpinBox()
        self.layout_weight.setRange(0.0, 2.0)
        self.layout_weight.setValue(0.35)
        self.layout_weight.setSingleStep(0.05)
        tab.form.addRow("Hard constraint", constraint_note)
        tab.form.addRow("", self.fixed_layout)
        tab.form.addRow("Number of people", self.cluster_count)
        tab.form.addRow("Distance threshold", self.distance_threshold)
        tab.form.addRow("Automatic k maximum", self.kmax)
        tab.form.addRow("Layout penalty weight", self.layout_weight)
        tab.run_button.clicked.connect(self.start_clustering)
        return tab

    def paths(self):
        return PipelinePaths(self.workspace_field.path())

    def frame_offset(self):
        return -1

    def start_job(self, tab, function, success):
        if self.thread is not None:
            QMessageBox.information(
                self, "Pipeline busy", "Wait for the current stage to finish."
            )
            return
        tab.log.clear()
        tab.status.setText("Running…")
        tab.run_button.setEnabled(False)
        self.stream = SignalStream()
        self.stream.text_written.connect(tab.append_log)
        self.thread = QThread(self)
        self.worker = Worker(function, self.stream)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(
            lambda result: self.job_succeeded(tab, success, result)
        )
        self.worker.failed.connect(lambda error: self.job_failed(tab, error))
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_job)
        self.thread.start()

    def job_succeeded(self, tab, callback, result):
        tab.status.setText("Complete")
        tab.run_button.setEnabled(True)
        callback(result)

    def job_failed(self, tab, error):
        tab.append_log(error)
        tab.status.setText("Failed")
        tab.run_button.setEnabled(True)
        QMessageBox.critical(
            self, "Stage failed", error.splitlines()[-1] if error else "Unknown error"
        )

    def cleanup_job(self):
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = self.thread = self.stream = None

    def start_tracking(self):
        config = TrackingConfig(
            video=self.video_field.path(),
            workspace=self.workspace_field.path(),
            detector=self.detector.text().strip(),
            reid=self.boxmot_reid.text().strip(),
            tracker=self.tracker.text().strip(),
            device=self.device.text().strip(),
        )
        self.start_job(
            self.tracking_tab, lambda: run_tracking(config), self.tracking_finished
        )

    def tracking_finished(self, paths):
        self.tracking_tab.preview.load(
            self.video_field.path(), paths.tracks, frame_offset=self.frame_offset()
        )
        self.tabs.setCurrentWidget(self.alignment_tab)

    def start_alignment(self):
        config = AlignmentConfig(
            video=self.video_field.path(),
            gaze_tsv=self.gaze_field.path(),
            workspace=self.workspace_field.path(),
            time_col=self.time_col.text().strip(),
            reuse_markers=self.reuse_markers.isChecked(),
        )
        self.start_job(
            self.alignment_tab, lambda: run_alignment(config), self.alignment_finished
        )

    def alignment_finished(self, paths):
        self.alignment_tab.preview.load(
            self.video_field.path(),
            paths.tracks,
            gaze_path=paths.aligned_gaze,
            frame_offset=self.frame_offset(),
        )
        self.tabs.setCurrentWidget(self.face_tab)

    def start_faces(self):
        config = FaceConfig(
            video=self.video_field.path(),
            workspace=self.workspace_field.path(),
            face_model=self.face_model.text().strip(),
            face_threshold=self.face_threshold.value(),
            max_samples=self.max_samples.value(),
            landmark_stride=self.landmark_stride.value(),
            reid_weights=self.reid_weights.path(),
            frame_offset=self.frame_offset(),
        )
        self.start_job(
            self.face_tab, lambda: run_face_preprocessing(config), self.faces_finished
        )

    def faces_finished(self, paths):
        self.face_tab.preview.load(
            self.video_field.path(),
            paths.tracks,
            landmarks_path=paths.raw_landmarks,
            gaze_path=paths.aligned_gaze,
            frame_offset=self.frame_offset(),
        )
        self.tabs.setCurrentWidget(self.clustering_tab)

    def start_clustering(self):
        n_clusters = self.cluster_count.value() or None
        threshold = self.distance_threshold.value() or None
        if n_clusters is not None and threshold is not None:
            QMessageBox.warning(
                self,
                "Conflicting options",
                "Choose either a fixed number of people or a distance threshold, not both.",
            )
            return
        config = ClusteringConfig(
            video=self.video_field.path(),
            workspace=self.workspace_field.path(),
            fixed_layout=self.fixed_layout.isChecked(),
            n_clusters=n_clusters,
            distance_threshold=threshold,
            kmax=self.kmax.value(),
            layout_weight=self.layout_weight.value(),
            frame_offset=self.frame_offset(),
        )
        self.start_job(
            self.clustering_tab,
            lambda: run_clustering(config),
            self.clustering_finished,
        )

    def clustering_finished(self, paths):
        self.clustering_tab.preview.load(
            self.video_field.path(),
            paths.clustered_tracks,
            person_labels=True,
            landmarks_path=paths.clustered_landmarks,
            gaze_path=paths.aligned_gaze,
            gaze_on_faces_path=paths.gaze_on_faces,
            frame_offset=self.frame_offset(),
        )

    def load_existing_workspace(self, show_message=False):
        paths = self.paths()
        video = self.video_field.path()
        if not video.is_file():
            if show_message:
                QMessageBox.warning(
                    self, "Video not found", f"Video not found: {video}"
                )
            return []
        if not paths.workspace.is_dir():
            if show_message:
                QMessageBox.warning(
                    self,
                    "Workspace not found",
                    f"Workspace directory not found: {paths.workspace}",
                )
            return []

        stages = [
            ("Tracking", self.tracking_tab),
            ("Gaze alignment", self.alignment_tab),
            ("Faces / ReID", self.face_tab),
            ("Clustering", self.clustering_tab),
        ]
        for _, tab in stages:
            tab.preview.reset("No cached output for this stage")
            tab.status.setText("Not available")

        loaded = []
        if paths.tracks.exists():
            self.tracking_tab.preview.load(
                video, paths.tracks, frame_offset=self.frame_offset()
            )
            self.tracking_tab.status.setText("Loaded from workspace")
            loaded.append("Tracking")
        if paths.tracks.exists() and paths.aligned_gaze.exists():
            self.alignment_tab.preview.load(
                video,
                paths.tracks,
                gaze_path=paths.aligned_gaze,
                frame_offset=self.frame_offset(),
            )
            self.alignment_tab.status.setText("Loaded from workspace")
            loaded.append("Gaze alignment")
        if paths.raw_landmarks.exists():
            self.face_tab.preview.load(
                video,
                paths.tracks,
                landmarks_path=paths.raw_landmarks,
                gaze_path=paths.aligned_gaze,
                frame_offset=self.frame_offset(),
            )
            self.face_tab.status.setText("Loaded from workspace")
            loaded.append("Faces / ReID")
        if paths.clustered_tracks.exists():
            self.clustering_tab.preview.load(
                video,
                paths.clustered_tracks,
                person_labels=True,
                landmarks_path=paths.clustered_landmarks,
                gaze_path=paths.aligned_gaze,
                gaze_on_faces_path=paths.gaze_on_faces,
                frame_offset=self.frame_offset(),
            )
            self.clustering_tab.status.setText("Loaded from workspace")
            loaded.append("Clustering")

        if show_message:
            detail = ", ".join(loaded) if loaded else "No completed stages found"
            QMessageBox.information(
                self,
                "Workspace loaded",
                f"Workspace: {paths.workspace}\nVideo: {video}\n\n{detail}",
            )
        return loaded

    def refresh_available_previews(self):
        self.load_existing_workspace(show_message=False)

    def closeEvent(self, event):
        if self.thread is not None:
            QMessageBox.warning(
                self, "Pipeline busy", "Wait for the running stage to finish."
            )
            event.ignore()
            return
        for preview in [
            self.tracking_tab.preview,
            self.alignment_tab.preview,
            self.face_tab.preview,
            self.clustering_tab.preview,
        ]:
            preview.close_video()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
