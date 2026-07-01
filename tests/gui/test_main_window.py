import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui import MainWindow


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
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _run_button(window, step_type):
    """The pipeline editor section's "Run" button for ``step_type``."""
    return next(
        s.run_button
        for s in window.pipeline_editor._sections
        if s.step_type is step_type
    )


def _pin_face_model(window, name="buffalo_l"):
    """Pin the face model to a small pack so tests don't pull the heavier default."""
    section = next(
        s for s in window.pipeline_editor._sections if s.step_type is FaceDetectionStep
    )
    section.form._widgets["model_name"].setCurrentText(name)


def test_run_button_disabled_until_video_loaded(window):
    assert not _run_button(window, ObjectTrackingStep).isEnabled()


def test_window_has_icon(window):
    assert not window.windowIcon().isNull()


def test_load_video_enables_run_button(window, data_dir):
    video = data_dir / "three-people.mp4"
    window._load_video(video)

    assert window.video_viewer.frame_count == 5
    assert _run_button(window, ObjectTrackingStep).isEnabled()
    assert not _run_button(window, BodyPoseStep).isEnabled()
    assert str(video) in window.file_label.text()


def test_run_object_tracking_populates_state_and_overlays(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    assert (
        window.video_viewer._overlay_items == []
    )  # nothing drawn before object tracking

    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window.video.data is not None, timeout=60000)
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)

    assert window.video.data["track_id"].nunique() == 3
    # Object tracking drives the video, so it ends on the last frame with that frame's
    # boxes drawn (3 people -> 3 rects + 3 labels).
    assert window.video_viewer.current_frame == window.video_viewer.frame_count - 1
    assert len(window.video_viewer._overlay_items) == 6
    # Controls are restored and the worker thread is cleaned up once done.
    assert _run_button(window, ObjectTrackingStep).isEnabled()
    assert window.video_viewer._play_button.isEnabled()
    assert not window.progress_bar.isVisible()
    assert not window.cancel_button.isVisible()


def test_face_button_disabled_until_object_tracking_done(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    # Face detection runs on tracked boxes, so it waits for object tracking.
    assert not _run_button(window, FaceDetectionStep).isEnabled()
    assert not _run_button(window, BodyPoseStep).isEnabled()

    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)
    assert _run_button(window, FaceDetectionStep).isEnabled()
    assert _run_button(window, BodyPoseStep).isEnabled()


def test_run_face_detection_populates_state_and_overlays(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window.video.data is not None, timeout=60000)
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)

    _pin_face_model(window)
    window._start_step(FaceDetectionStep)
    qtbot.waitUntil(
        lambda: window._thread is None and "face_score" in window.video.data.columns,
        timeout=120000,
    )

    data = window.video.data
    # The tracked rows are preserved and gain face columns.
    assert data["track_id"].nunique() == 3
    assert data["face_score"].notna().any()
    # Ends on the last frame drawing that frame's person boxes and faces.
    last = window.video_viewer.frame_count - 1
    assert window.video_viewer.current_frame == last
    assert window.video.faces_for_frame(last)
    assert _run_button(window, ObjectTrackingStep).isEnabled()
    assert _run_button(window, FaceDetectionStep).isEnabled()


def _experiment_folder(tmp_path, video, with_output):
    """A saved experiment folder for ``video``, optionally with a cached output."""
    experiment = Experiment(
        name="demo",
        inputs=[VideoInput(id="cam1", path=video)],
    )
    experiment.to_dir(tmp_path)
    if with_output:
        data = pd.DataFrame(
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
        output = experiment.output_path(experiment.inputs[0])
        output.parent.mkdir(parents=True, exist_ok=True)
        video_obj = Video()
        video_obj.set_data(data)
        video_obj.to_parquet(output)
    return tmp_path


def test_open_experiment_loads_video_and_cached_results(window, data_dir, tmp_path):
    folder = _experiment_folder(
        tmp_path, data_dir / "three-people.mp4", with_output=True
    )

    window._load_experiment(folder)

    # The experiment's video is opened and its cached tracking results loaded.
    assert window.video_viewer.frame_count == 5
    assert window.video.data["track_id"].nunique() == 3
    # Cached tracks are present, so the later passes are ready to run.
    assert _run_button(window, FaceDetectionStep).isEnabled()
    assert _run_button(window, BodyPoseStep).isEnabled()


def test_open_experiment_without_output_loads_video_only(window, data_dir, tmp_path):
    folder = _experiment_folder(
        tmp_path, data_dir / "three-people.mp4", with_output=False
    )

    window._load_experiment(folder)

    assert window.video_viewer.frame_count == 5
    assert window.video.data is None
    # No cached tracks, so face/pose wait until object tracking runs.
    assert not _run_button(window, FaceDetectionStep).isEnabled()
    assert not _run_button(window, BodyPoseStep).isEnabled()


def test_open_invalid_experiment_shows_error(qtbot, window, tmp_path, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: shown.setdefault("called", True),
    )

    window._load_experiment(tmp_path)  # empty folder, no experiment.yaml

    assert shown.get("called")
    assert window.video.data is None


def test_open_video_creates_single_input_experiment(window, data_dir):
    video = data_dir / "three-people.mp4"
    window._load_video(video)

    assert window.experiment is not None
    assert [i.path for i in window.experiment.inputs] == [video]
    # The pipeline editor seeds the mandatory tracking step; the rest are opt-in.
    assert [type(s) for s in window.experiment.steps] == [ObjectTrackingStep]
    assert window._experiment_dir is None
    assert window.save_action.isEnabled()


def test_running_object_tracking_records_step_in_experiment(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)

    assert [type(s) for s in window.experiment.steps] == [ObjectTrackingStep]


def test_new_experiment_resets_state(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)

    window._new_experiment()

    assert window.experiment is None
    assert window.video.data is None
    assert not _run_button(window, ObjectTrackingStep).isEnabled()
    assert not window.save_action.isEnabled()
    assert window.file_label.text() == "No file selected"


def test_save_writes_experiment_and_results(qtbot, window, data_dir, tmp_path):
    window._load_video(data_dir / "three-people.mp4")
    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)

    window._experiment_dir = tmp_path
    window._save_experiment()

    reloaded = Experiment.from_dir(tmp_path)
    assert [type(s) for s in reloaded.steps] == [ObjectTrackingStep]
    # Results are written beside the config and reload with the tracked boxes.
    output = reloaded.output_path(reloaded.inputs[0])
    assert Video.from_parquet(output).data["track_id"].nunique() == 3


def test_save_then_open_round_trips_through_the_window(
    qtbot, window, data_dir, tmp_path
):
    window._load_video(data_dir / "three-people.mp4")
    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)
    window._experiment_dir = tmp_path
    window._save_experiment()

    window._new_experiment()
    window._load_experiment(tmp_path)

    assert window.video.data["track_id"].nunique() == 3
    assert [type(s) for s in window.experiment.steps] == [ObjectTrackingStep]
    assert window._experiment_dir == tmp_path
    assert _run_button(window, FaceDetectionStep).isEnabled()


def test_load_experiment_public_entry_point(window, data_dir, tmp_path):
    folder = _experiment_folder(
        tmp_path, data_dir / "three-people.mp4", with_output=True
    )

    window.load_experiment(str(folder))

    assert window.video.data["track_id"].nunique() == 3
    assert window._experiment_dir == tmp_path


def test_title_shows_open_experiment_folder(qtbot, window, data_dir, tmp_path):
    folder = tmp_path / "boop"
    _experiment_folder(folder, data_dir / "three-people.mp4", with_output=True)

    assert window.windowTitle() == "body-eye-sync"  # nothing open yet

    window.load_experiment(folder)
    assert window.windowTitle() == "body-eye-sync :: [boop]"

    # Opening a loose video is an unsaved experiment, so the folder drops away.
    window._load_video(data_dir / "three-people.mp4")
    assert window.windowTitle() == "body-eye-sync"

    window._new_experiment()
    assert window.windowTitle() == "body-eye-sync"


def test_pipeline_editor_disabled_until_experiment(window, data_dir):
    assert not window.pipeline_editor.isEnabled()

    window._load_video(data_dir / "three-people.mp4")
    assert window.pipeline_editor.isEnabled()

    window._new_experiment()
    assert not window.pipeline_editor.isEnabled()


def test_editing_pipeline_updates_experiment(window, data_dir):
    from body_eye_sync.experiment.config import FaceDetectionStep

    window._load_video(data_dir / "three-people.mp4")
    assert [type(s) for s in window.experiment.steps] == [ObjectTrackingStep]

    # Enable the face-detection section via the editor.
    face_section = next(
        s for s in window.pipeline_editor._sections if s.step_type is FaceDetectionStep
    )
    face_section.setChecked(True)

    assert [type(s) for s in window.experiment.steps] == [
        ObjectTrackingStep,
        FaceDetectionStep,
    ]


def test_loaded_pipeline_populates_editor(qtbot, window, data_dir, tmp_path):
    from body_eye_sync.experiment.config import (
        Experiment as _Experiment,
        FaceDetectionStep,
        ObjectTrackingStep,
        VideoInput,
    )

    experiment = _Experiment(
        name="demo",
        inputs=[VideoInput(id="cam1", path=data_dir / "three-people.mp4")],
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(det_thresh=0.8),
    )
    experiment.to_dir(tmp_path)

    window.load_experiment(tmp_path)

    # The editor reflects the loaded pipeline, including the tuned argument.
    steps = {type(s): s for s in window.pipeline_editor.enabled_steps()}
    assert set(steps) == {ObjectTrackingStep, FaceDetectionStep}
    assert steps[FaceDetectionStep].det_thresh == 0.8


def test_editor_args_reach_the_tracking_worker(
    qtbot, window, data_dir, fast_object_tracking
):
    window._load_video(data_dir / "three-people.mp4")

    tracking = next(
        s for s in window.pipeline_editor._sections if s.step_type is ObjectTrackingStep
    )
    tracking.form._widgets["detector"].setCurrentText("yolov8m")

    window._start_step(ObjectTrackingStep)
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)

    # The detector the user chose in the editor is what the pipeline was called with.
    assert fast_object_tracking["kwargs"]["detector"] == "yolov8m"


def test_invalid_step_settings_abort_the_run(
    window, data_dir, monkeypatch, fast_object_tracking
):
    window._load_video(data_dir / "three-people.mp4")

    tracking = next(
        s for s in window.pipeline_editor._sections if s.step_type is ObjectTrackingStep
    )
    tracking.form._widgets["object_classes"].setText("not-a-number")

    shown = {}
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QMessageBox.critical",
        lambda *a, **k: shown.setdefault("called", True),
    )

    window._start_step(ObjectTrackingStep)

    # The invalid config is reported and no run is started.
    assert shown.get("called")
    assert window._thread is None
    assert "kwargs" not in fast_object_tracking
    assert window.video.data is None


def test_transport_disabled_while_object_tracking(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    assert window.video_viewer._play_button.isEnabled()

    window._start_step(ObjectTrackingStep)
    # The worker drives the frame during object tracking, so manual transport is locked.
    assert not window.video_viewer._play_button.isEnabled()
    assert not window.video_viewer._slider.isEnabled()

    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)
    assert window.video_viewer._play_button.isEnabled()


def test_step_run_button_click_starts_that_step(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")

    _run_button(window, ObjectTrackingStep).click()
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)

    assert window.video.data["track_id"].nunique() == 3


def test_run_all_chains_enabled_steps_in_order(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    face_section = next(
        s for s in window.pipeline_editor._sections if s.step_type is FaceDetectionStep
    )
    face_section.setChecked(True)
    _pin_face_model(window)

    window.pipeline_editor.run_all_button.click()
    qtbot.waitUntil(
        lambda: window._thread is None and "face_score" in window.video.data.columns,
        timeout=120000,
    )

    data = window.video.data
    # Object tracking ran first, so face detection had tracked boxes to use.
    assert data["track_id"].nunique() == 3
    assert data["face_score"].notna().any()
    assert window._pending_steps == []


def test_run_all_stops_the_chain_on_failure(qtbot, window, data_dir, monkeypatch):
    window._load_video(data_dir / "three-people.mp4")
    face_section = next(
        s for s in window.pipeline_editor._sections if s.step_type is FaceDetectionStep
    )
    face_section.setChecked(True)

    def failing_detect_tracklets(video_path, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - never reached, makes this a generator

    monkeypatch.setattr(
        "body_eye_sync.pipeline.object_tracking.detect_tracklets",
        failing_detect_tracklets,
    )
    # Avoid blocking on the modal failure dialog ``_on_failed`` shows.
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QMessageBox.exec", lambda self: None
    )

    window.pipeline_editor.run_all_button.click()
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)

    # The failed tracking step aborted the chain, so face detection never ran.
    assert window._pending_steps == []
    assert window.video.data is None
