from types import SimpleNamespace

import numpy as np
import pytest

from body_eye_sync.experiment import run as run_module
from body_eye_sync.experiment.config import (
    AudioInput,
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
from body_eye_sync.experiment.run import run_experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult
from body_eye_sync.pipeline.body_pose import BodyPose, PoseFrameResult
from body_eye_sync.pipeline.object_tracking import BoundingBox


def _frame(frame_idx, *boxes):
    rows = [[x1, y1, x2, y2, tid, conf, 0, 0] for x1, y1, x2, y2, tid, conf in boxes]
    tracks = np.array(rows) if rows else np.empty((0, 8))
    return SimpleNamespace(frame_idx=frame_idx, tracks=tracks)


def _face_result(frame_idx, *faces):
    boxes = [
        FaceBox(BoundingBox(x1, y1, x2, y2, tid), score, [(x1, y1)] * 5)
        for tid, x1, y1, x2, y2, score in faces
    ]
    return FaceFrameResult(frame_idx, boxes)


def _pose_result(frame_idx, *poses):
    bodies = [
        BodyPose(BoundingBox(x1, y1, x2, y2, tid), score, [(x1, y1, 0.9)] * 17)
        for tid, x1, y1, x2, y2, score in poses
    ]
    return PoseFrameResult(frame_idx, bodies)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace the heavy detect_* functions with lightweight fakes.

    Returns a dict of stage name to the list of calls made to it, in order, each
    recording the video it ran on and the kwargs it was given, so tests can
    assert step args are forwarded correctly.
    """
    calls: dict[str, list[dict]] = {"tracking": [], "face": [], "pose": []}

    def fake_tracklets(video_path, **kwargs):
        calls["tracking"].append({"video": video_path, **kwargs})
        return iter([_frame(1, (0.0, 0.0, 5.0, 5.0, 1, 0.9))])

    def fake_faces(video_path, boxes_by_frame, **kwargs):
        calls["face"].append({"video": video_path, "boxes": boxes_by_frame, **kwargs})
        return iter([_face_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.9))])

    def fake_poses(video_path, boxes_by_frame, **kwargs):
        calls["pose"].append({"video": video_path, "boxes": boxes_by_frame, **kwargs})
        return iter([_pose_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.8))])

    monkeypatch.setattr(run_module, "detect_tracklets", fake_tracklets)
    monkeypatch.setattr(run_module, "detect_faces", fake_faces)
    monkeypatch.setattr(run_module, "detect_body_poses", fake_poses)
    return calls


def _full_pipeline(**overrides):
    kwargs = dict(
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(),
        body_pose=BodyPoseStep(),
    )
    kwargs.update(overrides)
    return VideoPipeline(**kwargs)


def _experiment(video_file, **overrides):
    kwargs = dict(
        name="demo",
        glasses_videos=[GlassesVideoInput(id="cam1", path=video_file)],
        pipeline=Pipeline(glasses_video=_full_pipeline(), fixed_video=_full_pipeline()),
    )
    kwargs.update(overrides)
    return Experiment(ExperimentConfig(**kwargs), video_file.parent)


def test_run_writes_parquet_per_input(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()

    results = run_experiment(_experiment(video_file))

    assert results == {"cam1": tmp_path / "outputs" / "cam1.parquet"}
    data = Video.from_parquet(results["cam1"]).data
    # One tracked box in frame 0, with face and pose columns merged on.
    assert len(data) == 1
    assert data["face_score"].notna().all()
    assert data["pose_score"].notna().all()


def test_run_forwards_step_args(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    exp = _experiment(
        video_file,
        pipeline=Pipeline(
            glasses_video=_full_pipeline(
                object_tracking=ObjectTrackingStep(
                    detector="yolov8m", object_classes=[0, 32]
                ),
                face_detection=FaceDetectionStep(det_thresh=0.7),
                body_pose=BodyPoseStep(conf=0.4),
            )
        ),
    )

    run_experiment(exp)

    tracking, face, pose = (stub_pipeline[k][0] for k in ("tracking", "face", "pose"))
    assert tracking["detector"] == "yolov8m"
    assert tracking["object_classes"] == [0, 32]
    assert face["det_thresh"] == 0.7
    assert pose["conf"] == 0.4
    # Face/pose passes receive the tracked boxes.
    assert set(face["boxes"]) == {0}
    # Device/providers are auto-detected inside the detect_* functions, not
    # forwarded from run.
    assert "device" not in tracking
    assert "providers" not in face


def test_each_video_type_runs_its_own_pipeline_block(tmp_path, stub_pipeline):
    glasses_file = tmp_path / "glasses.mp4"
    glasses_file.touch()
    room_file = tmp_path / "room.mp4"
    room_file.touch()
    exp = _experiment(
        glasses_file,
        fixed_videos=[FixedVideoInput(id="room", path=room_file)],
        pipeline=Pipeline(
            glasses_video=_full_pipeline(
                object_tracking=ObjectTrackingStep(detector="yolov8m")
            ),
            # The room camera skips face detection and tracks with its own model.
            fixed_video=_full_pipeline(
                object_tracking=ObjectTrackingStep(detector="yolo26x"),
                face_detection=None,
            ),
        ),
    )

    results = run_experiment(exp)

    assert set(results) == {"cam1", "room"}
    assert [c["video"] for c in stub_pipeline["tracking"]] == [glasses_file, room_file]
    assert [c["detector"] for c in stub_pipeline["tracking"]] == ["yolov8m", "yolo26x"]
    # Only the glasses video's block enables face detection.
    assert [c["video"] for c in stub_pipeline["face"]] == [glasses_file]
    assert [c["video"] for c in stub_pipeline["pose"]] == [glasses_file, room_file]


def test_audio_inputs_are_skipped(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    audio_file = tmp_path / "p1.wav"
    audio_file.touch()
    exp = _experiment(
        video_file,
        audio=[AudioInput(id="mic1", path=audio_file, glasses_video="cam1")],
    )

    results = run_experiment(exp)

    # Audio has no pipeline stages yet, so it produces no output at all.
    assert set(results) == {"cam1"}
    assert not (tmp_path / "outputs" / "mic1.parquet").exists()


def test_existing_output_is_skipped_without_force(tmp_path, monkeypatch):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    exp = _experiment(video_file)
    destination = exp.output_path(exp.glasses_videos[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"stale")

    def boom(*args, **kwargs):
        raise AssertionError("pipeline should not run when the output exists")

    monkeypatch.setattr(run_module, "detect_tracklets", boom)

    results = run_experiment(exp)
    assert results["cam1"] == destination
    assert destination.read_bytes() == b"stale"  # untouched


def test_force_reruns_and_overwrites(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    exp = _experiment(video_file)
    destination = exp.output_path(exp.glasses_videos[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"stale")

    results = run_experiment(exp, force=True)
    assert Video.from_parquet(results["cam1"]).data is not None


def test_runs_a_loaded_experiment(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    _experiment(video_file).save()

    results = run_experiment(Experiment.load(tmp_path))
    assert results["cam1"] == tmp_path / "outputs" / "cam1.parquet"
    assert results["cam1"].exists()


def test_missing_video_raises(tmp_path, stub_pipeline):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError, match="cam1"):
        run_experiment(_experiment(missing))
