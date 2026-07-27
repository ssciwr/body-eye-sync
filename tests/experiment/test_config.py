import pytest
from pydantic import ValidationError

from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    AudioInput,
    AudioPipeline,
    BodyPoseStep,
    ExperimentConfig,
    FaceDetectionStep,
    FixedVideoInput,
    GlassesVideoInput,
    ObjectTrackingStep,
    Pipeline,
    VideoPipeline,
)


def _video_pipeline(**overrides):
    kwargs = dict(
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(),
        body_pose=BodyPoseStep(),
    )
    kwargs.update(overrides)
    return VideoPipeline(**kwargs)


def _config(**overrides):
    kwargs = dict(
        glasses_videos=[
            GlassesVideoInput(
                id="cam1", path="videos/session1.mp4", gaze_path="videos/session1.tsv"
            )
        ],
        pipeline=Pipeline(
            glasses_video=_video_pipeline(), fixed_video=_video_pipeline()
        ),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_default_models_are_recent_medium():
    # The product defaults are recent medium-size models (a good speed/accuracy
    # balance), deliberately heavier than the pipeline functions' lightweight
    # fallbacks. Tests that run real models pin the smallest models explicitly.
    pipeline = _config().pipeline.glasses_video
    assert (pipeline.object_tracking.detector, pipeline.object_tracking.reid) == (
        "yolo26m",
        "osnet_x1_0_msmt17",
    )
    assert pipeline.object_tracking.tracker == "botsort"
    assert pipeline.object_tracking.object_classes == [0]
    assert (
        pipeline.face_detection.model_name,
        pipeline.face_detection.det_size,
        pipeline.face_detection.det_thresh,
    ) == ("antelopev2", 640, 0.5)
    assert (pipeline.body_pose.model_name, pipeline.body_pose.conf) == (
        "yolo26m-pose.pt",
        0.25,
    )


def test_object_tracking_defaults_when_omitted():
    cfg = ExperimentConfig(
        glasses_videos=[GlassesVideoInput(id="cam1", path="v.mp4", gaze_path="v.tsv")]
    )
    pipeline = cfg.pipeline.glasses_video
    assert isinstance(pipeline.object_tracking, ObjectTrackingStep)
    assert pipeline.face_detection is None
    assert pipeline.body_pose is None


def test_steps_lists_present_stages_tracking_first():
    pipeline = _video_pipeline(face_detection=None)
    assert [type(s) for s in pipeline.steps] == [ObjectTrackingStep, BodyPoseStep]


def test_tracking_only_pipeline_is_valid():
    pipeline = _video_pipeline(face_detection=None, body_pose=None)
    assert [type(s) for s in pipeline.steps] == [ObjectTrackingStep]


def test_audio_pipeline_has_no_steps_yet():
    assert AudioPipeline().steps == []


def test_version_defaults_to_current():
    assert _config().version == CURRENT_VERSION


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        ObjectTrackingStep(detecter="yolov8n")  # typo


def test_a_glasses_video_needs_its_gaze_file():
    # The pipeline has no use for one without the other, so it is not optional.
    with pytest.raises(ValidationError, match="gaze_path"):
        GlassesVideoInput(id="cam1", path="v.mp4")


def test_time_offset_defaults_to_zero():
    assert (
        GlassesVideoInput(id="cam1", path="v.mp4", gaze_path="v.tsv").time_offset == 0.0
    )
    assert AudioInput(id="mic1", path="a.wav").time_offset == 0.0


def test_inputs_are_stored_in_typed_lists():
    cfg = _config(
        fixed_videos=[FixedVideoInput(id="room", path="room.mp4")],
        audio=[AudioInput(id="mic1", path="a.wav")],
    )
    assert [video.id for video in cfg.glasses_videos] == ["cam1"]
    assert [video.id for video in cfg.fixed_videos] == ["room"]
    assert [audio.id for audio in cfg.audio] == ["mic1"]


def test_audio_only_experiment_is_valid():
    cfg = ExperimentConfig(
        glasses_videos=[], audio=[AudioInput(id="mic1", path="a.wav")]
    )
    assert cfg.glasses_videos == []
    assert cfg.fixed_videos == []
    assert [audio.id for audio in cfg.audio] == ["mic1"]


def test_video_pipeline_blocks_are_independent():
    cfg = _config(fixed_videos=[FixedVideoInput(id="room", path="room.mp4")])
    cfg.pipeline.fixed_video.face_detection = None
    cfg.pipeline.fixed_video.object_tracking.detector = "yolo26x"
    assert cfg.pipeline.glasses_video.face_detection is not None
    assert cfg.pipeline.glasses_video.object_tracking.detector == "yolo26m"


def test_an_experiment_with_no_inputs_is_valid():
    # What a new experiment in the GUI is, until input files are added to it.
    cfg = _config(glasses_videos=[])
    assert cfg.glasses_videos == []
    assert cfg.fixed_videos == []
    assert cfg.audio == []


def test_duplicate_input_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate input ids"):
        _config(
            glasses_videos=[
                GlassesVideoInput(id="cam1", path="a.mp4", gaze_path="a.tsv"),
                GlassesVideoInput(id="cam1", path="b.mp4", gaze_path="b.tsv"),
            ]
        )


def test_input_ids_must_be_unique_across_types():
    with pytest.raises(ValidationError, match="duplicate input ids"):
        _config(audio=[AudioInput(id="cam1", path="a.wav")])


def test_audio_may_reference_a_glasses_video():
    cfg = _config(audio=[AudioInput(id="mic1", path="a.wav", glasses_video="cam1")])
    assert cfg.audio[0].glasses_video == "cam1"


def test_audio_referencing_an_unknown_glasses_video_rejected():
    with pytest.raises(ValidationError, match="unknown glasses video ids"):
        _config(audio=[AudioInput(id="mic1", path="a.wav", glasses_video="nope")])


def test_audio_referencing_a_fixed_video_rejected():
    # Only glasses videos identify a participant, so a fixed camera is not a
    # valid target even though its id exists.
    with pytest.raises(ValidationError, match="unknown glasses video ids"):
        _config(
            fixed_videos=[FixedVideoInput(id="room", path="room.mp4")],
            audio=[AudioInput(id="mic1", path="a.wav", glasses_video="room")],
        )
