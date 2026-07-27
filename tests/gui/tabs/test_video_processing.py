import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    ExperimentConfig,
    FaceDetectionStep,
    FixedVideoInput,
    GlassesVideoInput,
    ObjectTrackingStep,
    Pipeline,
    VideoPipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import FixedVideo
from body_eye_sync.gui.tabs.video_processing import VideoProcessingTab


def _tracks_frame(frame_idx):
    # BoxMOT layout: x1, y1, x2, y2, id, conf, cls, det_ind.
    # Boxes cover the three visible people in tests/data/three-people.mp4.
    return SimpleNamespace(
        frame_idx=frame_idx,
        tracks=np.array(
            [
                [0.0, 55.0, 155.0, 310.0, 1, 0.9, 0, 0],
                [310.0, 40.0, 460.0, 310.0, 2, 0.9, 0, 1],
                [135.0, 35.0, 340.0, 310.0, 3, 0.9, 0, 2],
            ]
        ),
    )


@pytest.fixture(autouse=True)
def fast_object_tracking(monkeypatch):
    calls = {}

    def detect_tracklets(video_path, **kwargs):
        calls["kwargs"] = kwargs
        for frame_idx in range(1, 6):
            time.sleep(0.01)
            yield _tracks_frame(frame_idx)

    monkeypatch.setattr(
        "body_eye_sync.pipeline.object_tracking.detect_tracklets", detect_tracklets
    )
    return calls


@pytest.fixture
def experiment(data_dir):
    """An experiment with one glasses video, and nothing computed yet."""
    return Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1",
                    path=data_dir / "three-people.mp4",
                    gaze_path=data_dir / "three-people.tsv",
                )
            ],
        )
    )


@pytest.fixture
def tab(qtbot, experiment):
    tab = VideoProcessingTab(experiment)
    qtbot.addWidget(tab)
    return tab


@pytest.fixture
def empty_tab(qtbot):
    """The tab as it is for an experiment with no inputs at all."""
    tab = VideoProcessingTab(Experiment(ExperimentConfig()))
    qtbot.addWidget(tab)
    return tab


def _run_button(tab, step_type):
    """The pipeline editor section's "Run" button for ``step_type``."""
    return next(
        s.run_button for s in tab.pipeline_editor._sections if s.step_type is step_type
    )


def _section(tab, step_type):
    return next(s for s in tab.pipeline_editor._sections if s.step_type is step_type)


def _pin_face_model(tab, name="buffalo_l"):
    """Pin the face model to a small pack so tests don't pull the heavier default."""
    _section(tab, FaceDetectionStep).form._widgets["model_name"].setCurrentText(name)


def _tracked_frame():
    return pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "track_id": [1, 2, 3],
            "x1": [0.0, 310.0, 135.0],
            "y1": [55.0, 40.0, 35.0],
            "x2": [155.0, 460.0, 340.0],
            "y2": [310.0, 310.0, 310.0],
            "conf": [0.9, 0.9, 0.9],
        }
    )


def test_without_inputs_nothing_can_be_run(empty_tab):
    assert empty_tab.video() is None
    assert empty_tab.video_selector.count() == 0
    assert not empty_tab.video_selector.isEnabled()
    assert not empty_tab.pipeline_editor.isEnabled()
    assert not _run_button(empty_tab, ObjectTrackingStep).isEnabled()


def test_the_experiments_video_is_shown_and_ready_to_track(tab):
    assert tab.video_selector.currentText() == "cam1 (glasses)"
    assert tab.video_viewer.frame_count == 5
    assert tab.pipeline_editor.isEnabled()
    assert _run_button(tab, ObjectTrackingStep).isEnabled()
    # Face/pose run over tracked boxes, so they wait for object tracking.
    assert not _run_button(tab, FaceDetectionStep).isEnabled()
    assert not _run_button(tab, BodyPoseStep).isEnabled()


def test_run_object_tracking_populates_state_and_overlays(qtbot, tab):
    assert tab.video_viewer._overlay_items == []  # nothing drawn before tracking

    tab._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: tab.video().data is not None, timeout=60000)
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=5000)

    assert tab.video().data["track_id"].nunique() == 3
    # Object tracking drives the video, so it ends on the last frame with that
    # frame's boxes drawn (3 people -> 3 rects + 3 labels).
    assert tab.video_viewer.current_frame == tab.video_viewer.frame_count - 1
    assert len(tab.video_viewer._overlay_items) == 6
    # Controls are restored and the worker thread is cleaned up once done.
    assert _run_button(tab, ObjectTrackingStep).isEnabled()
    assert _run_button(tab, FaceDetectionStep).isEnabled()
    assert _run_button(tab, BodyPoseStep).isEnabled()
    assert tab.video_viewer._play_button.isEnabled()
    assert not tab.progress_bar.isVisibleTo(tab)
    assert not tab.cancel_button.isVisibleTo(tab)


def test_run_face_detection_populates_state_and_overlays(qtbot, tab):
    tab._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: tab.video().data is not None, timeout=60000)
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=5000)

    _pin_face_model(tab)
    tab._start_step(FaceDetectionStep)
    qtbot.waitUntil(
        lambda: not tab.is_busy() and "face_score" in tab.video().data.columns,
        timeout=120000,
    )

    data = tab.video().data
    # The tracked rows are preserved and gain face columns.
    assert data["track_id"].nunique() == 3
    assert data["face_score"].notna().any()
    # Real face embeddings were captured (K=32 default) for later clustering.
    face_emb = tab.video().face_embeddings
    assert face_emb is not None
    assert face_emb["embedding"].iloc[0].dtype == np.float16
    # Ends on the last frame drawing that frame's person boxes and faces.
    last = tab.video_viewer.frame_count - 1
    assert tab.video_viewer.current_frame == last
    assert tab.video().faces_for_frame(last)


def test_cached_results_are_shown_and_unlock_the_later_steps(qtbot, experiment):
    experiment.glasses_videos[0].set_data(_tracked_frame())

    tab = VideoProcessingTab(experiment)
    qtbot.addWidget(tab)

    assert _run_button(tab, FaceDetectionStep).isEnabled()
    assert _run_button(tab, BodyPoseStep).isEnabled()
    # The first frame's cached boxes are drawn: 3 rects + 3 labels.
    assert len(tab.video_viewer._overlay_items) == 6


def test_a_fixed_video_edits_the_fixed_video_pipeline_block(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
            ],
        )
    )
    tab = VideoProcessingTab(experiment)
    qtbot.addWidget(tab)

    assert isinstance(tab.video(), FixedVideo)
    assert tab.video_selector.currentText() == "room (fixed)"
    assert tab._pipeline() is experiment.pipeline.fixed_video


def test_choosing_another_video_shows_it_and_its_pipeline_block(qtbot, tab, data_dir):
    tab.experiment.add_fixed_video(
        FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
    )
    tab.refresh()
    assert [
        tab.video_selector.itemText(i) for i in range(tab.video_selector.count())
    ] == ["cam1 (glasses)", "room (fixed)"]

    tab.video_selector.setCurrentIndex(1)

    assert tab.video() is tab.experiment.fixed_videos[0]
    assert tab._pipeline() is tab.experiment.pipeline.fixed_video
    assert tab.video_viewer.frame_count == 5


def test_refresh_keeps_showing_the_same_video(tab, data_dir):
    tab.experiment.add_fixed_video(
        FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
    )
    tab.refresh()
    tab.video_selector.setCurrentIndex(1)
    tab.video_viewer.set_frame(2)

    # An unrelated input was added elsewhere, so the tab re-reads the experiment.
    tab.experiment.add_fixed_video(
        FixedVideoInput(id="room2", path=data_dir / "three-people.mp4")
    )
    tab.refresh()

    assert tab.video() is tab.experiment.fixed_videos[0]
    # The video was not reloaded, so playback stayed where it was.
    assert tab.video_viewer.current_frame == 2


def test_inputs_that_changed_during_a_run_are_re_read_when_it_ends(
    qtbot, tab, data_dir
):
    tab._start_step(ObjectTrackingStep)
    assert tab.is_busy()

    # An input added while the run is on cannot go in yet: the run drives the
    # viewer and holds the video it writes into.
    tab.experiment.add_fixed_video(
        FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
    )
    tab.refresh()
    assert tab.video_selector.count() == 1

    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)

    # Once it is over the tab re-reads the experiment, so the change is not lost.
    assert [
        tab.video_selector.itemText(i) for i in range(tab.video_selector.count())
    ] == ["cam1 (glasses)", "room (fixed)"]
    assert tab.video() is tab.experiment.glasses_videos[0]
    assert tab.video().data is not None


def test_removing_the_shown_video_falls_back_to_the_first(tab, data_dir):
    tab.experiment.add_fixed_video(
        FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
    )
    tab.refresh()
    tab.video_selector.setCurrentIndex(1)

    tab.experiment.remove_input(tab.experiment.fixed_videos[0])
    tab.refresh()

    assert tab.video() is tab.experiment.glasses_videos[0]


def test_a_video_file_that_cannot_be_opened_is_reported_and_tried_again(
    qtbot, tmp_path, data_dir
):
    path = tmp_path / "arriving.mp4"
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1", path=path, gaze_path=path.with_suffix(".tsv")
                )
            ]
        )
    )
    tab = VideoProcessingTab(experiment)  # its file is not there yet
    qtbot.addWidget(tab)
    messages = []
    tab.status_message.connect(messages.append)

    tab.refresh()  # the failure is reported whenever the video is (re)loaded

    assert tab.video_viewer.frame_count == 0
    assert messages and messages[0].startswith("Could not open video")

    # A failure does not count as a load, so the video is picked up once its
    # file can be opened rather than staying blank for good.
    path.write_bytes((data_dir / "three-people.mp4").read_bytes())
    tab.refresh()

    assert tab.video_viewer.frame_count == 5


def test_set_experiment_switches_to_the_new_ones_videos(tab, data_dir):
    other = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
            ],
        )
    )

    tab.set_experiment(other)

    assert tab.video() is other.fixed_videos[0]
    assert tab.video_selector.currentText() == "room (fixed)"


def test_set_experiment_without_videos_clears_the_viewer(tab):
    tab.set_experiment(Experiment(ExperimentConfig()))

    assert tab.video() is None
    assert tab.video_viewer.frame_count == 0
    assert tab.video_viewer._overlay_items == []
    assert not tab.pipeline_editor.isEnabled()


def test_editing_the_pipeline_updates_the_experiment(tab):
    assert [type(s) for s in tab.experiment.pipeline.glasses_video.steps] == [
        ObjectTrackingStep
    ]

    _section(tab, FaceDetectionStep).setChecked(True)

    assert [type(s) for s in tab.experiment.pipeline.glasses_video.steps] == [
        ObjectTrackingStep,
        FaceDetectionStep,
    ]


def test_a_loaded_pipeline_populates_the_editor(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1",
                    path=data_dir / "three-people.mp4",
                    gaze_path=data_dir / "three-people.tsv",
                )
            ],
            pipeline=Pipeline(
                glasses_video=VideoPipeline(
                    object_tracking=ObjectTrackingStep(),
                    face_detection=FaceDetectionStep(det_thresh=0.8),
                )
            ),
        )
    )
    tab = VideoProcessingTab(experiment)
    qtbot.addWidget(tab)

    steps = {type(s): s for s in tab.pipeline_editor.enabled_steps()}
    assert set(steps) == {ObjectTrackingStep, FaceDetectionStep}
    assert steps[FaceDetectionStep].det_thresh == 0.8


def test_editor_args_reach_the_tracking_worker(qtbot, tab, fast_object_tracking):
    _section(tab, ObjectTrackingStep).form._widgets["detector"].setCurrentText(
        "yolov8m"
    )

    tab._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)

    # The detector the user chose in the editor is what the pipeline was called with.
    assert fast_object_tracking["kwargs"]["detector"] == "yolov8m"


def test_invalid_step_settings_abort_the_run(tab, monkeypatch, fast_object_tracking):
    _section(tab, ObjectTrackingStep).form._widgets["object_classes"].setText("nope")
    shown = {}
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.video_processing.QMessageBox.critical",
        lambda *a, **k: shown.setdefault("called", True),
    )

    tab._start_step(ObjectTrackingStep)

    # The invalid config is reported and no run is started.
    assert shown.get("called")
    assert not tab.is_busy()
    assert "kwargs" not in fast_object_tracking
    assert tab.video().data is None


def test_the_video_is_locked_while_a_step_runs(qtbot, tab):
    assert tab.video_viewer._play_button.isEnabled()
    busy = []
    tab.busy_changed.connect(busy.append)

    tab._start_step(ObjectTrackingStep)

    # The worker drives the frame during a run, so manual transport is locked,
    # as is switching to another video.
    assert not tab.video_viewer._play_button.isEnabled()
    assert not tab.video_viewer._slider.isEnabled()
    assert not tab.video_selector.isEnabled()
    assert tab.progress_bar.isVisibleTo(tab)
    assert busy == [True]

    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)
    assert tab.video_viewer._play_button.isEnabled()
    assert tab.video_selector.isEnabled()
    assert busy == [True, False]


def test_a_run_in_progress_ignores_a_refresh(qtbot, tab, data_dir):
    tab._start_step(ObjectTrackingStep)
    tab.experiment.add_fixed_video(
        FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
    )

    tab.refresh()

    # The new input waits: re-listing the videos mid-run would pull the video
    # being written to out from under the worker.
    assert tab.video_selector.count() == 1
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)


def test_step_run_button_click_starts_that_step(qtbot, tab):
    _run_button(tab, ObjectTrackingStep).click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)

    assert tab.video().data["track_id"].nunique() == 3


def test_finishing_a_step_reports_it(qtbot, tab):
    messages = []
    tab.status_message.connect(messages.append)

    _run_button(tab, ObjectTrackingStep).click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)

    assert messages == ["Object tracking finished: 3 tracklets, 15 detections"]


def test_run_all_chains_enabled_steps_in_order(qtbot, tab):
    _section(tab, FaceDetectionStep).setChecked(True)
    _pin_face_model(tab)

    tab.pipeline_editor.run_all_button.click()
    qtbot.waitUntil(
        lambda: not tab.is_busy() and "face_score" in tab.video().data.columns,
        timeout=120000,
    )

    data = tab.video().data
    # Object tracking ran first, so face detection had tracked boxes to use.
    assert data["track_id"].nunique() == 3
    assert data["face_score"].notna().any()
    assert tab._pending_steps == []


def test_run_all_stops_the_chain_on_failure(qtbot, tab, monkeypatch):
    _section(tab, FaceDetectionStep).setChecked(True)

    def failing_detect_tracklets(video_path, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - never reached, makes this a generator

    monkeypatch.setattr(
        "body_eye_sync.pipeline.object_tracking.detect_tracklets",
        failing_detect_tracklets,
    )
    # Avoid blocking on the modal failure dialog ``_on_failed`` shows.
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.video_processing.QMessageBox.exec", lambda self: None
    )

    tab.pipeline_editor.run_all_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=60000)

    # The failed tracking step aborted the chain, so face detection never ran.
    assert tab._pending_steps == []
    assert tab.video().data is None


def test_shutdown_cancels_a_running_step(qtbot, tab):
    tab._start_step(ObjectTrackingStep)

    tab.shutdown()

    assert tab._thread is None or not tab._thread.is_alive()
