import pytest
from pydantic import ValidationError

from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)


def _experiment(**overrides):
    kwargs = dict(
        name="demo",
        inputs=[VideoInput(id="cam1", path="videos/session1.mp4")],
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(),
        body_pose=BodyPoseStep(),
    )
    kwargs.update(overrides)
    return Experiment(**kwargs)


def test_default_models_are_recent_medium():
    # The product defaults are recent medium-size models (a good speed/accuracy
    # balance), deliberately heavier than the pipeline functions' lightweight
    # fallbacks. Tests that run real models pin the smallest models explicitly.
    exp = _experiment()
    assert (exp.object_tracking.detector, exp.object_tracking.reid) == (
        "yolo26m",
        "osnet_x1_0_msmt17",
    )
    assert exp.object_tracking.tracker == "botsort"
    assert exp.object_tracking.object_classes == [0]
    assert (
        exp.face_detection.model_name,
        exp.face_detection.det_size,
        exp.face_detection.det_thresh,
    ) == ("antelopev2", 640, 0.5)
    assert (exp.body_pose.model_name, exp.body_pose.conf) == ("yolo26m-pose.pt", 0.25)


def test_object_tracking_defaults_when_omitted():
    exp = Experiment(name="demo", inputs=[VideoInput(id="cam1", path="v.mp4")])
    assert isinstance(exp.object_tracking, ObjectTrackingStep)
    assert exp.face_detection is None
    assert exp.body_pose is None


def test_steps_lists_present_stages_tracking_first():
    exp = _experiment(face_detection=None)
    assert [type(s) for s in exp.steps] == [ObjectTrackingStep, BodyPoseStep]


def test_version_defaults_to_current():
    assert _experiment().version == CURRENT_VERSION


def test_yaml_round_trip_preserves_everything(tmp_path):
    original = _experiment(
        object_tracking=ObjectTrackingStep(detector="yolov8m", object_classes=[0, 32]),
        face_detection=FaceDetectionStep(det_thresh=0.7),
        body_pose=None,
    )
    path = tmp_path / "experiment.yaml"
    original.to_yaml(path)

    loaded = Experiment.from_yaml(path)
    assert loaded.model_dump() == original.model_dump()


def test_from_yaml_loads_named_steps(tmp_path):
    path = tmp_path / "experiment.yaml"
    _experiment(body_pose=None).to_yaml(path)

    loaded = Experiment.from_yaml(path)
    assert isinstance(loaded.object_tracking, ObjectTrackingStep)
    assert isinstance(loaded.face_detection, FaceDetectionStep)
    assert loaded.body_pose is None


def test_relative_paths_resolve_against_experiment_dir(tmp_path):
    path = tmp_path / "experiment.yaml"
    _experiment().to_yaml(path)

    loaded = Experiment.from_yaml(path)
    resolved = loaded.resolved_input_path(loaded.inputs[0])
    assert resolved == (tmp_path / "videos" / "session1.mp4").resolve()


def test_absolute_paths_pass_through(tmp_path):
    absolute = tmp_path / "clip.mp4"
    exp = _experiment(inputs=[VideoInput(id="cam1", path=absolute)])
    assert exp.resolved_input_path(exp.inputs[0]) == absolute


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        ObjectTrackingStep(detecter="yolov8n")  # typo


def test_empty_inputs_rejected():
    with pytest.raises(ValidationError, match="no inputs"):
        _experiment(inputs=[])


def test_duplicate_input_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate input ids"):
        _experiment(
            inputs=[
                VideoInput(id="cam1", path="a.mp4"),
                VideoInput(id="cam1", path="b.mp4"),
            ]
        )


def test_tracking_only_pipeline_is_valid():
    exp = _experiment(face_detection=None, body_pose=None)
    assert [type(s) for s in exp.steps] == [ObjectTrackingStep]


def test_to_dir_and_from_dir_round_trip(tmp_path):
    folder = tmp_path / "experiment"
    original = _experiment()

    config_file = original.to_dir(folder)
    assert config_file == folder / "experiment.yaml"

    loaded = Experiment.from_dir(folder)
    assert loaded.model_dump() == original.model_dump()
    assert loaded.base_dir == folder


def test_output_path_is_under_outputs_dir(tmp_path):
    exp = _experiment()
    exp.to_dir(tmp_path / "experiment")
    spec = exp.inputs[0]
    assert exp.output_dir == tmp_path / "experiment" / "outputs"
    assert exp.output_path(spec) == tmp_path / "experiment" / "outputs" / "cam1.parquet"


def test_newer_file_version_rejected(tmp_path):
    path = tmp_path / "experiment.yaml"
    _experiment().to_yaml(path)
    text = path.read_text().replace(
        f"version: {CURRENT_VERSION}", f"version: {CURRENT_VERSION + 1}"
    )
    path.write_text(text)

    with pytest.raises(ValueError, match="newer than supported"):
        Experiment.from_yaml(path)
