from types import SimpleNamespace

import numpy as np
import pytest

from body_eye_sync.experiment import run as run_module
from body_eye_sync.experiment.config import (
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.run import read_provenance, run_experiment
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

    Returns a dict recording the kwargs each stage was called with, so tests can
    assert step args and runtime options are forwarded correctly.
    """
    calls: dict[str, dict] = {}

    def fake_tracklets(video_path, **kwargs):
        calls["tracking"] = kwargs
        return iter([_frame(1, (0.0, 0.0, 5.0, 5.0, 1, 0.9))])

    def fake_faces(video_path, boxes_by_frame, **kwargs):
        calls["face"] = {"boxes": boxes_by_frame, **kwargs}
        return iter([_face_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.9))])

    def fake_poses(video_path, boxes_by_frame, **kwargs):
        calls["pose"] = {"boxes": boxes_by_frame, **kwargs}
        return iter([_pose_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.8))])

    monkeypatch.setattr(run_module, "detect_tracklets", fake_tracklets)
    monkeypatch.setattr(run_module, "detect_faces", fake_faces)
    monkeypatch.setattr(run_module, "detect_body_poses", fake_poses)
    return calls


def _experiment(video_file, **overrides):
    kwargs = dict(
        name="demo",
        inputs=[VideoInput(id="cam1", path=video_file)],
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(),
        body_pose=BodyPoseStep(),
    )
    kwargs.update(overrides)
    return Experiment(**kwargs)


def test_run_writes_parquet_per_input(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    out_dir = tmp_path / "out"

    results = run_experiment(_experiment(video_file), output_dir=out_dir)

    assert results == {"cam1": out_dir / "cam1.parquet"}
    data = Video.from_parquet(results["cam1"]).data
    # One tracked box in frame 0, with face and pose columns merged on.
    assert len(data) == 1
    assert data["face_score"].notna().all()
    assert data["pose_score"].notna().all()


def test_run_forwards_step_args_and_runtime_options(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    exp = _experiment(
        video_file,
        object_tracking=ObjectTrackingStep(detector="yolov8m", object_classes=[0, 32]),
        face_detection=FaceDetectionStep(det_thresh=0.7),
        body_pose=BodyPoseStep(conf=0.4),
    )

    run_experiment(
        exp,
        output_dir=tmp_path / "out",
        device="cpu",
        providers=["CPUExecutionProvider"],
    )

    assert stub_pipeline["tracking"]["detector"] == "yolov8m"
    assert stub_pipeline["tracking"]["object_classes"] == [0, 32]
    assert stub_pipeline["tracking"]["device"] == "cpu"
    assert stub_pipeline["face"]["det_thresh"] == 0.7
    assert stub_pipeline["face"]["providers"] == ["CPUExecutionProvider"]
    assert stub_pipeline["pose"]["conf"] == 0.4
    assert stub_pipeline["pose"]["device"] == "cpu"
    # Face/pose passes receive the tracked boxes.
    assert set(stub_pipeline["face"]["boxes"]) == {0}


def test_provenance_is_stamped(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()

    results = run_experiment(
        _experiment(video_file), output_dir=tmp_path / "out", device="cpu"
    )

    prov = read_provenance(results["cam1"])
    assert prov["experiment"] == "demo"
    assert prov["input_id"] == "cam1"
    assert prov["device"] == "cpu"
    assert set(prov["pipeline"]) == {"object_tracking", "face_detection", "body_pose"}
    assert prov["pipeline"]["object_tracking"]["detector"] == "yolo26m"
    assert "body-eye-sync" in prov["versions"]
    assert "created" in prov


def test_existing_output_is_skipped_without_force(tmp_path, monkeypatch):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    destination = out_dir / "cam1.parquet"
    destination.write_bytes(b"stale")

    def boom(*args, **kwargs):
        raise AssertionError("pipeline should not run when the output exists")

    monkeypatch.setattr(run_module, "detect_tracklets", boom)

    results = run_experiment(_experiment(video_file), output_dir=out_dir)
    assert results["cam1"] == destination
    assert destination.read_bytes() == b"stale"  # untouched


def test_force_reruns_and_overwrites(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "cam1.parquet").write_bytes(b"stale")

    results = run_experiment(_experiment(video_file), output_dir=out_dir, force=True)
    assert Video.from_parquet(results["cam1"]).data is not None


def test_default_output_dir_beside_experiment_file(tmp_path, stub_pipeline):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    yaml_path = tmp_path / "experiment.yaml"
    _experiment(video_file).to_yaml(yaml_path)

    results = run_experiment(Experiment.from_yaml(yaml_path))
    assert results["cam1"] == tmp_path / "outputs" / "cam1.parquet"
    assert results["cam1"].exists()


def test_missing_video_raises(tmp_path, stub_pipeline):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError, match="cam1"):
        run_experiment(_experiment(missing), output_dir=tmp_path / "out")
