import pytest
from pydantic import ValidationError

from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    BodyPoseStep,
    ExperimentConfig,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)


def _config(**overrides):
    kwargs = dict(
        name="demo",
        inputs=[VideoInput(id="cam1", path="videos/session1.mp4")],
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(),
        body_pose=BodyPoseStep(),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_default_models_are_recent_medium():
    # The product defaults are recent medium-size models (a good speed/accuracy
    # balance), deliberately heavier than the pipeline functions' lightweight
    # fallbacks. Tests that run real models pin the smallest models explicitly.
    cfg = _config()
    assert (cfg.object_tracking.detector, cfg.object_tracking.reid) == (
        "yolo26m",
        "osnet_x1_0_msmt17",
    )
    assert cfg.object_tracking.tracker == "botsort"
    assert cfg.object_tracking.object_classes == [0]
    assert (
        cfg.face_detection.model_name,
        cfg.face_detection.det_size,
        cfg.face_detection.det_thresh,
    ) == ("antelopev2", 640, 0.5)
    assert (cfg.body_pose.model_name, cfg.body_pose.conf) == ("yolo26m-pose.pt", 0.25)


def test_object_tracking_defaults_when_omitted():
    cfg = ExperimentConfig(name="demo", inputs=[VideoInput(id="cam1", path="v.mp4")])
    assert isinstance(cfg.object_tracking, ObjectTrackingStep)
    assert cfg.face_detection is None
    assert cfg.body_pose is None


def test_steps_lists_present_stages_tracking_first():
    cfg = _config(face_detection=None)
    assert [type(s) for s in cfg.steps] == [ObjectTrackingStep, BodyPoseStep]


def test_tracking_only_pipeline_is_valid():
    cfg = _config(face_detection=None, body_pose=None)
    assert [type(s) for s in cfg.steps] == [ObjectTrackingStep]


def test_version_defaults_to_current():
    assert _config().version == CURRENT_VERSION


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        ObjectTrackingStep(detecter="yolov8n")  # typo


def test_empty_inputs_rejected():
    with pytest.raises(ValidationError, match="no inputs"):
        _config(inputs=[])


def test_duplicate_input_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate input ids"):
        _config(
            inputs=[
                VideoInput(id="cam1", path="a.mp4"),
                VideoInput(id="cam1", path="b.mp4"),
            ]
        )
