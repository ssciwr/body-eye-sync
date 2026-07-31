"""Video processing tab: run the pipeline over one video input and watch it.

One video input is shown at a time, chosen from the experiment's video inputs.
The pipeline editor beside the viewer edits the pipeline block belonging to that
input's type, and its "Run" buttons run the steps in a background thread, with
the results drawn over the video as they arrive.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pydantic import ValidationError

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoPipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import PipelineEditor, VideoViewer
from body_eye_sync.gui.workers import (
    BodyPoseWorker,
    FaceDetectionWorker,
    ObjectTrackingWorker,
)


@dataclass
class _StepRunner:
    """How to run one pipeline step: its worker and the plumbing it needs."""

    worker_cls: type
    # if the step is ready to run (e.g. face/pose need tracked boxes).
    ready: Callable[[Video], bool]
    # clear any previous results for this step
    begin: Callable[[Video, int], None]
    # the slot the worker's live frames are drawn with, in addition to the shared one.
    live_frame_slot: Callable
    on_finished: Callable[[], None]


class VideoProcessingTab(BaseTab):
    """Show one of the experiment's videos and run its pipeline over it."""

    title = "Video processing"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)

        self._thread: threading.Thread | None = None
        self._worker: (
            ObjectTrackingWorker | FaceDetectionWorker | BodyPoseWorker | None
        ) = None
        self._in_setup = False
        #: Remaining step types queued by "Run all"; consumed one at a time as
        #: each step finishes, so later steps see earlier steps' results.
        self._pending_steps: list[type] = []
        #: The experiment's video inputs, in the order the chooser lists them.
        self._videos: list[Video] = []

        self.video_selector = QComboBox()
        self.video_selector.currentIndexChanged.connect(self._on_video_selected)

        self.video_viewer = VideoViewer()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_run)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Video:"))
        top_bar.addWidget(self.video_selector, stretch=1)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.progress_bar, stretch=1)
        bottom_bar.addWidget(self.cancel_button)

        viewer_layout = QVBoxLayout()
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addLayout(top_bar)
        viewer_layout.addWidget(self.video_viewer, stretch=1)
        viewer_layout.addLayout(bottom_bar)
        viewer_side = QWidget()
        viewer_side.setLayout(viewer_layout)

        # The pipeline editor is the authority for the shown video's pipeline
        # block. The splitter lets the user give it as much room as they want.
        self.pipeline_editor = PipelineEditor()
        self.pipeline_editor.changed.connect(self._on_pipeline_edited)
        self.pipeline_editor.run_requested.connect(self._start_step)
        self.pipeline_editor.run_all_requested.connect(self._start_run_all)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(viewer_side)
        self.splitter.addWidget(self.pipeline_editor)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)

        # How to run each step: its worker, readiness check and viewer plumbing.
        # Built last since it closes over ``self.video_viewer``.
        self._step_runners: dict[type, _StepRunner] = {
            ObjectTrackingStep: _StepRunner(
                worker_cls=ObjectTrackingWorker,
                ready=lambda video: video.video_path is not None,
                begin=lambda video, k: video.begin_object_tracking(k),
                live_frame_slot=self.video_viewer.show_live_frame,
                on_finished=self._on_finished,
            ),
            FaceDetectionStep: _StepRunner(
                worker_cls=FaceDetectionWorker,
                ready=lambda video: video.data is not None,
                begin=lambda video, k: video.begin_face_detection(k),
                live_frame_slot=self.video_viewer.show_live_face_frame,
                on_finished=self._on_face_finished,
            ),
            BodyPoseStep: _StepRunner(
                worker_cls=BodyPoseWorker,
                ready=lambda video: video.data is not None,
                begin=lambda video, k: video.begin_body_pose_detection(k),
                live_frame_slot=self.video_viewer.show_live_pose_frame,
                on_finished=self._on_pose_finished,
            ),
        }
        self.refresh()

    def refresh(self) -> None:
        """Re-list the experiment's videos, keeping the shown one if it is still there."""
        if self._thread is not None:
            # A run drives the viewer and the video it writes into; leave it be.
            return
        shown = self.video()
        self._videos = [*self.experiment.glasses_videos, *self.experiment.fixed_videos]
        self.video_selector.blockSignals(True)
        self.video_selector.clear()
        for video in self._videos:
            kind = "glasses" if isinstance(video, GlassesVideo) else "fixed"
            self.video_selector.addItem(f"{video.id} ({kind})")
        index = next(
            (i for i, video in enumerate(self._videos) if video is shown),
            0 if self._videos else -1,
        )
        self.video_selector.setCurrentIndex(index)
        self.video_selector.blockSignals(False)
        self.video_selector.setEnabled(bool(self._videos))
        self._show_selected_video()

    def video(self) -> Video | None:
        """The video input being shown, or ``None`` if the experiment has none."""
        index = self.video_selector.currentIndex()
        if 0 <= index < len(self._videos):
            return self._videos[index]
        return None

    def is_busy(self) -> bool:
        """Whether a pipeline step is currently running."""
        return self._thread is not None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _on_video_selected(self, _index: int) -> None:
        self._show_selected_video()

    def _show_selected_video(self) -> None:
        """Load the chosen video into the viewer and bind the editor to it."""
        video = self.video()
        # The viewer says what it holds, so a video it could not open is tried
        # again next time rather than being left blank for good.
        if video is not self.video_viewer.video:
            if video is None:
                self.video_viewer.clear()
            else:
                try:
                    self.video_viewer.load(video)
                except OSError as exc:
                    self.video_viewer.clear()
                    self.status_message.emit(f"Could not open video: {exc}")
        self.video_viewer.refresh_overlays()
        self._bind_editor_to_video()
        self._update_step_availability()

    def _pipeline(self) -> VideoPipeline | None:
        """The pipeline block for the shown video's input type, or ``None``.

        Each input type has its own block, so the editor edits the one belonging
        to the video being shown.
        """
        video = self.video()
        if video is None:
            return None
        if isinstance(video, GlassesVideo):
            return self.experiment.pipeline.glasses_video
        return self.experiment.pipeline.fixed_video

    def _bind_editor_to_video(self) -> None:
        """Populate the pipeline editor from the shown video (or disable it)."""
        pipeline = self._pipeline()
        if pipeline is None:
            self.pipeline_editor.reset()
            self.pipeline_editor.setEnabled(False)
            return
        self.pipeline_editor.setEnabled(True)
        self.pipeline_editor.set_from(pipeline)

    def _on_pipeline_edited(self) -> None:
        """Adopt the editor's pipeline as the experiment's, when it is valid."""
        pipeline = self._pipeline()
        if pipeline is None:
            return
        try:
            self.pipeline_editor.apply_to(pipeline)
        except (ValidationError, ValueError):
            self.status_message.emit("Pipeline has invalid settings; not applied")
            return
        self.experiment_changed.emit()

    def _update_step_availability(self) -> None:
        """Enable each step's "Run" button (and "Run all") once its inputs are ready.

        Object tracking needs a video; later passes run on tracked boxes, so
        they wait for object tracking's results. Whether a *run* is currently
        in progress is handled separately, by disabling the whole pipeline
        editor (see ``_set_running``).
        """
        video = self.video()
        has_video = video is not None and video.video_path is not None
        has_tracks = video is not None and video.data is not None
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
        video = self.video()
        runner = self._step_runners[step_type]
        if video is None or not runner.ready(video):
            self._pending_steps = []
            return
        step = self._step_config(step_type)
        if step is None:
            self._pending_steps = []
            return

        # Discards that step's previous results; keeps everything else (e.g. a
        # face/pose pass keeps the tracked boxes it runs over). The embedding
        # budget comes from the step config, so the GUI and CLI behave identically.
        runner.begin(video, getattr(step, "embeddings_per_track", 0))
        self._begin_run()

        self._worker = runner.worker_cls(video, step)
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
        data = self.video().data
        self.status_message.emit(
            f"Object tracking finished: {data['track_id'].nunique()} tracklets, "
            f"{len(data)} detections"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot()
    def _on_face_finished(self) -> None:
        data = self.video().data
        n_faces = int(data["face_score"].notna().sum())
        self.status_message.emit(
            f"Face detection finished: {n_faces} faces over {len(data)} detections"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot()
    def _on_pose_finished(self) -> None:
        data = self.video().data
        n_poses = int(data["pose_score"].notna().sum())
        self.status_message.emit(
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
        self.status_message.emit(f"{self._worker.operation_name} cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            # background thread has reported back; drop our references to it.
            self._thread = None
            self._worker = None
            # Re-read the experiment: refresh() bows out while a run is on, so
            # this is where any change made to the inputs meanwhile is picked up.
            self.refresh()
            # However it ended, the run changed the video's results.
            self.experiment_changed.emit()
        self.video_selector.setEnabled(not running and bool(self._videos))
        self.pipeline_editor.setEnabled(not running and self.video() is not None)
        self.video_viewer.enable_controls(not running)
        self.progress_bar.setVisible(running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        # The window locks the actions that would pull the experiment away.
        self.busy_changed.emit(running)
