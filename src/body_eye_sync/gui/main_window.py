from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Callable

from qtpy.QtCore import Qt, Slot
from qtpy.QtGui import QAction, QIcon, QKeySequence
from qtpy.QtWidgets import (
    QDockWidget,
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

from pydantic import ValidationError

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.body_pose_worker import BodyPoseWorker
from body_eye_sync.gui.face_detection_worker import FaceDetectionWorker
from body_eye_sync.gui.object_tracking_worker import ObjectTrackingWorker
from body_eye_sync.gui.pipeline_editor import PipelineEditor
from body_eye_sync.gui.video_viewer import VideoViewer

#: Base window title; the open experiment folder is appended when there is one.
_BASE_TITLE = "body-eye-sync"


@dataclass
class _StepRunner:
    """How to run one pipeline step: its worker and the window plumbing it needs."""

    worker_cls: type
    #: Whether the step's inputs are ready (e.g. face/pose need tracked boxes).
    ready: Callable[[], bool]
    #: Clears any previous results for this step and prepares the video for it.
    begin: Callable[[], None]
    #: Slot the worker's live frames are drawn with, in addition to the shared one.
    live_frame_slot: Callable
    on_finished: Callable[[], None]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_BASE_TITLE)
        with as_file(files(__package__) / "resources" / "icon.ico") as icon_path:
            self.setWindowIcon(QIcon(str(icon_path)))

        # The window owns the experiment (what to run) and the video (its
        # results). The experiment is None until a video is opened or an existing
        # experiment folder is loaded; ``_experiment_dir`` is the save location,
        # None until it has been saved/opened.
        self.experiment: Experiment | None = None
        self._experiment_dir: Path | None = None
        self.video = Video()
        self._thread: threading.Thread | None = None
        self._worker: (
            ObjectTrackingWorker | FaceDetectionWorker | BodyPoseWorker | None
        ) = None
        self._in_setup = False
        #: Remaining step types queued by "Run all"; consumed one at a time as
        #: each step finishes, so later steps see earlier steps' results.
        self._pending_steps: list[type] = []

        self._build_menu_bar()

        self.open_button = QPushButton("Open video…")
        self.open_button.clicked.connect(self._choose_video)

        self.file_label = QLabel("No file selected")

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
        bottom_bar.addWidget(self.progress_bar, stretch=1)
        bottom_bar.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.video_viewer, stretch=1)
        layout.addLayout(bottom_bar)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # The pipeline editor is the authority for the experiment's pipeline. It
        # lives in a movable dock so it is easy to relocate later.
        self.pipeline_editor = PipelineEditor()
        self.pipeline_editor.setEnabled(False)
        self.pipeline_editor.changed.connect(self._on_pipeline_edited)
        self.pipeline_editor.run_requested.connect(self._start_step)
        self.pipeline_editor.run_all_requested.connect(self._start_run_all)
        self.pipeline_dock = QDockWidget("Pipeline", self)
        self.pipeline_dock.setWidget(self.pipeline_editor)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.pipeline_dock)

        # How to run each step: its worker, readiness check and window plumbing.
        # Built last since it closes over ``self.video_viewer``.
        self._step_runners: dict[type, _StepRunner] = {
            ObjectTrackingStep: _StepRunner(
                worker_cls=ObjectTrackingWorker,
                ready=lambda: self.video.video_path is not None,
                begin=self.video.begin_object_tracking,
                live_frame_slot=self.video_viewer.show_live_frame,
                on_finished=self._on_finished,
            ),
            FaceDetectionStep: _StepRunner(
                worker_cls=FaceDetectionWorker,
                ready=lambda: self.video.data is not None,
                begin=self.video.begin_face_detection,
                live_frame_slot=self.video_viewer.show_live_face_frame,
                on_finished=self._on_face_finished,
            ),
            BodyPoseStep: _StepRunner(
                worker_cls=BodyPoseWorker,
                ready=lambda: self.video.data is not None,
                begin=self.video.begin_body_pose_detection,
                live_frame_slot=self.video_viewer.show_live_pose_frame,
                on_finished=self._on_pose_finished,
            ),
        }
        self._update_step_availability()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.new_action = QAction("&New", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self._new_experiment)
        file_menu.addAction(self.new_action)

        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._choose_experiment)
        file_menu.addAction(self.open_action)

        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._save_experiment)
        self.save_action.setEnabled(False)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

    def _new_experiment(self) -> None:
        """Reset to an empty state, discarding the current experiment/results."""
        if self._thread is not None:
            return
        self.video.clear()
        self.experiment = None
        self._experiment_dir = None
        self._update_title()
        self.file_label.setText("No file selected")
        self.video_viewer.refresh_overlays()
        self.video_viewer.enable_controls(False)
        self._update_step_availability()
        self.save_action.setEnabled(False)
        self._bind_editor_to_experiment()

    def _save_experiment(self) -> None:
        """Write the experiment (and any computed results) to its folder."""
        if self.experiment is None:
            return
        if self._experiment_dir is None:
            folder = QFileDialog.getExistingDirectory(self, "Save experiment to folder")
            if not folder:
                return
            self._experiment_dir = Path(folder)
            self._update_title()
        self.experiment.to_dir(self._experiment_dir)
        if self.video.data is not None:
            output = self.experiment.output_path(self.experiment.inputs[0])
            output.parent.mkdir(parents=True, exist_ok=True)
            self.video.to_parquet(output)
        self.statusBar().showMessage(f"Saved experiment to {self._experiment_dir}")

    def _update_title(self) -> None:
        """Show the open experiment folder in the title bar, if any."""
        if self._experiment_dir is not None:
            self.setWindowTitle(f"{_BASE_TITLE} :: [{self._experiment_dir.name}]")
        else:
            self.setWindowTitle(_BASE_TITLE)

    def _bind_editor_to_experiment(self) -> None:
        """Populate the pipeline editor from the experiment (or disable it)."""
        if self.experiment is None:
            self.pipeline_editor.reset()
            self.pipeline_editor.setEnabled(False)
            return
        self.pipeline_editor.setEnabled(True)
        self.pipeline_editor.set_from(self.experiment)

    def _on_pipeline_edited(self) -> None:
        """Adopt the editor's pipeline as the experiment's, when it is valid."""
        if self.experiment is None:
            return
        try:
            self.pipeline_editor.apply_to(self.experiment)
        except (ValidationError, ValueError):
            self.statusBar().showMessage("Pipeline has invalid settings; not applied")

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
        # Opening a video starts a fresh, unsaved single-input experiment.
        self.experiment = Experiment(
            name=path.stem or "experiment",
            inputs=[VideoInput(id=path.stem or "video", path=path)],
        )
        self._experiment_dir = None
        self._update_title()
        self._bind_editor_to_experiment()
        self.save_action.setEnabled(True)
        # A fresh video has no object tracking results yet, so later passes wait.
        self._update_step_availability()

    def load_experiment(self, folder: str | Path) -> None:
        """Open an experiment folder, as the File ▸ Open action does.

        Public entry point for launching the GUI on an experiment directly.
        Invalid folders are reported to the user, not raised.
        """
        self._load_experiment(Path(folder))

    def _choose_experiment(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open experiment folder")
        if folder:
            self._load_experiment(Path(folder))

    def _load_experiment(self, folder: Path) -> None:
        try:
            experiment = Experiment.from_dir(folder)
        except (OSError, ValueError, ValidationError) as exc:
            QMessageBox.critical(self, "Could not open experiment", str(exc))
            return

        # The GUI shows a single video, so open the experiment's first input and
        # its cached results (if any); the rest are still produced by the CLI.
        spec = experiment.inputs[0]
        video_path = experiment.resolved_input_path(spec)
        if not video_path.exists():
            QMessageBox.critical(self, "Video not found", str(video_path))
            return
        self._load_video(video_path)

        # _load_video started a fresh experiment for the video; adopt the loaded
        # one (with its pipeline) and remember the folder as the save location.
        self.experiment = experiment
        self._experiment_dir = Path(folder)
        self._update_title()
        self._bind_editor_to_experiment()

        output = experiment.output_path(spec)
        if output.exists():
            self.video.set_data(Video.from_parquet(output).data)
            self.video_viewer.refresh_overlays()
            self._update_step_availability()
            self.statusBar().showMessage(f"Loaded cached results from {output}")

    def _update_step_availability(self) -> None:
        """Enable each step's "Run" button (and "Run all") once its inputs are ready.

        Object tracking needs a video; later passes run on tracked boxes, so
        they wait for object tracking's results. Whether a *run* is currently
        in progress is handled separately, by disabling the whole pipeline
        editor (see ``_set_running``).
        """
        has_video = self.video.video_path is not None
        has_tracks = self.video.data is not None
        self.pipeline_editor.set_run_enabled(ObjectTrackingStep, has_video)
        self.pipeline_editor.set_run_enabled(FaceDetectionStep, has_tracks)
        self.pipeline_editor.set_run_enabled(BodyPoseStep, has_tracks)
        self.pipeline_editor.set_run_all_enabled(has_video)

    def _step_config(self, step_type):
        """The editor's validated config for a step, or None (with an alert)."""
        try:
            return self.pipeline_editor.config_for(step_type)
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return None

    @Slot(object)
    def _start_step(self, step_type: type) -> None:
        """Run one pipeline step, using the editor's current arguments for it."""
        if self._thread is not None:
            return
        runner = self._step_runners[step_type]
        if not runner.ready():
            self._pending_steps = []
            return
        step = self._step_config(step_type)
        if step is None:
            self._pending_steps = []
            return

        # Discards that step's previous results; keeps everything else (e.g. a
        # face/pose pass keeps the tracked boxes it runs over).
        runner.begin()
        self._begin_run()

        self._worker = runner.worker_cls(self.video, step)
        self._worker.new_frame.connect(self._on_new_frame)
        self._worker.new_frame.connect(runner.live_frame_slot)
        self._worker.finished.connect(runner.on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _start_run_all(self) -> None:
        """Run every enabled pipeline step in order, one after another."""
        if self._thread is not None:
            return
        try:
            steps = self.pipeline_editor.enabled_steps()
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self._pending_steps = [type(step) for step in steps]
        self._continue_run_all()

    def _continue_run_all(self) -> None:
        """Start the next step queued by "Run all", if any are left."""
        if self._pending_steps:
            self._start_step(self._pending_steps.pop(0))

    def _begin_run(self) -> None:
        """Shared start-up for object tracking and later detection runs."""
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
        self._continue_run_all()

    @Slot()
    def _on_face_finished(self) -> None:
        data = self.video.data
        n_faces = int(data["face_score"].notna().sum())
        self.statusBar().showMessage(
            f"Face detection finished: {n_faces} faces over {len(data)} detections"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot()
    def _on_pose_finished(self) -> None:
        data = self.video.data
        n_poses = int(data["pose_score"].notna().sum())
        self.statusBar().showMessage(
            f"Body pose detection finished: {n_poses} poses over {len(data)} detections"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        # A failure stops a "Run all" chain rather than pressing on regardless.
        self._pending_steps = []
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(f"{self._worker.operation_name} failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        # Cancelling one step cancels the rest of a "Run all" chain too.
        self._pending_steps = []
        self.statusBar().showMessage(f"{self._worker.operation_name} cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            # background thread has reported back; drop our references to it.
            self._thread = None
            self._worker = None
            self.video_viewer.refresh_overlays()
            self._update_step_availability()
        self.open_button.setEnabled(not running)
        self.new_action.setEnabled(not running)
        self.open_action.setEnabled(not running)
        self.save_action.setEnabled(not running and self.experiment is not None)
        self.pipeline_editor.setEnabled(not running and self.experiment is not None)
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
