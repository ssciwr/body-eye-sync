"""Serialisable definition of an experiment: its inputs and the pipeline to run."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The on-disk format version. Only needs to be bumped for non-backward-compatible changes, e.g. a new required field.
CURRENT_VERSION = 1


def validate_input_id(input_id: str) -> str:
    """Return an input id, having checked it can be used as a filename."""
    if not input_id:
        raise ValueError("input id cannot be empty")
    if any(char in input_id for char in ("/", "\\")) or input_id in (".", ".."):
        raise ValueError(f"input id cannot be used as a filename: {input_id!r}")
    return input_id


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeShift(_Model):
    """A stretch of a recording that was never written.

    Some devices stall briefly and carry on without it, so everything after
    sits earlier on their own clock than it does in the room. ``at`` is where
    that happens on the recording's own clock and ``seconds`` is how much is
    missing, which is added to the offset from there on.
    """

    at: float = Field(
        description="Where the loss falls on the recording's own clock, in seconds."
    )
    seconds: float = Field(gt=0, description="How much content is missing there.")


class _Input(_Model):
    """Fields shared by every experiment input, whatever its type.

    ``path`` may be relative; it is resolved against the experiment folder by the
    runtime :class:`Experiment`. Each input's type is given by which list of
    :class:`ExperimentConfig` it appears in, so the specs carry no ``kind`` tag.
    """

    id: str
    path: Path
    time_offset: float = Field(
        0.0,
        description=(
            "Seconds to add to this input's own clock to place it on the shared "
            "experiment timeline."
        ),
    )

    time_scale: float = Field(
        1.0,
        gt=0,
        description=(
            "Rate of experiment time per second of this input's own clock. "
            "Values slightly different from one compensate independent device clocks."
        ),
    )

    time_shifts: list[TimeShift] = Field(
        default_factory=list,
        description=("Content the recording lost partway through, if any."),
    )

    @field_validator("id")
    @classmethod
    def _check_id(cls, input_id: str) -> str:
        return validate_input_id(input_id)


class GlassesVideoInput(_Input):
    """Video and gaze data recorded by a participant's glasses-mounted camera."""

    gaze_path: Path = Field(
        description="Gaze samples recorded alongside this video, as a TSV file."
    )


class FixedVideoInput(_Input):
    """Video recorded by a camera at a fixed position in the room."""


class AudioInput(_Input):
    """Audio recorded on its own device, e.g. a directional microphone.

    The video inputs carry their own audio, so this is for separately recorded
    audio only.
    """

    glasses_video: str | None = Field(
        None,
        description=(
            "Optional id of the glasses video worn by the participant this "
            "recording captures"
        ),
    )


class ObjectTrackingStep(_Model):
    """Object detection + ReID tracking. Fields mirror ``detect_tracklets``.

    ``choices`` in a field's ``json_schema_extra`` are suggested values the GUI
    offers in an editable combobox; a custom value is still allowed.
    """

    detector: str = Field(
        "yolo26m",
        description="Object detector model.",
        json_schema_extra={
            "choices": [
                "yolo26n",
                "yolo26s",
                "yolo26m",
                "yolo26l",
                "yolo26x",
            ]
        },
    )
    reid: str = Field(
        "osnet_x1_0_msmt17",
        description="Re-identification model used to keep track ids stable.",
        json_schema_extra={
            "choices": [
                "osnet_x0_25_msmt17",
                "osnet_x0_5_msmt17",
                "osnet_x0_75_msmt17",
                "osnet_x1_0_msmt17",
                "osnet_ain_x1_0_msmt17",
                "mobilenetv2_x1_0_msmt17",
                "mobilenetv2_x1_4_msmt17",
                "resnet50_msmt17",
                "clip_market1501",
                "clip_duke",
            ]
        },
    )
    tracker: str = Field(
        "botsort",
        description="Multi-object tracking algorithm.",
        json_schema_extra={
            "choices": [
                "botsort",
                "bytetrack",
                "ocsort",
                "deepocsort",
                "hybridsort",
                "strongsort",
                "imprassoc",
                "boosttrack",
            ]
        },
    )
    object_classes: list[int] = Field(
        default=[0],
        description="COCO class ids to detect and track (0 = person).",
    )
    embeddings_per_track: int = Field(
        32,
        ge=0,
        description=(
            "Number of best body-appearance (ReID) embeddings to keep per tracklet"
        ),
    )


class FaceDetectionStep(_Model):
    """Per-box face detection. Fields mirror ``detect_faces``."""

    model_name: str = Field(
        "antelopev2",
        description="InsightFace model pack.",
        json_schema_extra={
            "choices": [
                "antelopev2",
                "buffalo_l",
                "buffalo_m",
                "buffalo_s",
                "buffalo_sc",
            ]
        },
    )
    det_size: int = Field(
        640, ge=64, le=2048, description="Detector input size in pixels."
    )
    det_thresh: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum face detection confidence."
    )
    embeddings_per_track: int = Field(
        32,
        ge=0,
        description=("Number of best face embeddings to keep per tracklet"),
    )


class BodyPoseStep(_Model):
    """Per-box body-pose detection. Fields mirror ``detect_body_poses``."""

    model_name: str = Field(
        "yolo26m-pose.pt",
        description="Ultralytics YOLO pose weights.",
        json_schema_extra={
            "choices": [
                "yolo26n-pose.pt",
                "yolo26s-pose.pt",
                "yolo26m-pose.pt",
                "yolo26l-pose.pt",
                "yolo26x-pose.pt",
            ]
        },
    )
    conf: float = Field(
        0.25, ge=0.0, le=1.0, description="Minimum pose detection confidence."
    )


class DiarizationStep(_Model):
    """Speaker diarization -- who spoke when. Fields mirror ``diarize``."""

    segmentation_model: str = Field(
        "sherpa-onnx-pyannote-segmentation-3-0",
        description="sherpa-onnx speaker segmentation model.",
        json_schema_extra={
            "choices": [
                "sherpa-onnx-pyannote-segmentation-3-0",
                "sherpa-onnx-reverb-diarization-v1",
                "sherpa-onnx-reverb-diarization-v2",
            ]
        },
    )
    embedding_model: str = Field(
        "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        description="Speaker embedding model used to cluster turns into speakers.",
        json_schema_extra={
            "choices": [
                "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
                "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
                "wespeaker_en_voxceleb_CAM++_LM.onnx",
                "wespeaker_en_voxceleb_resnet152_LM.onnx",
                "nemo_en_titanet_small.onnx",
                "nemo_en_titanet_large.onnx",
            ]
        },
    )
    num_speakers: int = Field(
        -1,
        ge=-1,
        description=(
            "Upper bound on the number of speakers, or -1 to use the threshold instead."
        ),
    )
    threshold: float = Field(
        0.5,
        gt=0.0,
        description=(
            "Distance below which two speech turns count as the same speaker, "
            "used when num_speakers is -1. Lower values find more speakers."
        ),
    )
    min_duration_on: float = Field(
        0.3, ge=0.0, description="Drop speech turns shorter than this many seconds."
    )
    min_duration_off: float = Field(
        0.5,
        ge=0.0,
        description="Bridge pauses shorter than this many seconds within a turn.",
    )
    embeddings_per_speaker: int = Field(
        32,
        ge=0,
        description=(
            "Number of best voice embeddings to keep per speaker, ranked by "
            "turn duration, for relating speakers across inputs later."
        ),
    )


class TranscriptionStep(_Model):
    """Speech transcription. Fields mirror ``transcribe``."""

    model_name: str = Field(
        "large-v3-turbo",
        description="Whisper model, run through faster-whisper.",
        json_schema_extra={
            "choices": [
                "tiny",
                "base",
                "small",
                "medium",
                "large-v3",
                "large-v3-turbo",
                "distil-large-v3",
            ]
        },
    )
    language: str | None = Field(
        None,
        description=(
            "ISO 639-1 language code of the recording, e.g. 'de'. Leave unset to "
            "detect it from the first 30 seconds."
        ),
    )
    beam_size: int = Field(5, ge=1, description="Decoding beam width.")
    vad_filter: bool = Field(
        True,
        description=(
            "Skip silent stretches, which speeds up the pass and suppresses text "
            "invented over silence."
        ),
    )


# A pipeline stage for type hints
StepSpec = Union[
    ObjectTrackingStep,
    FaceDetectionStep,
    BodyPoseStep,
    DiarizationStep,
    TranscriptionStep,
]


class _StepPipeline(_Model):
    """A set of pipeline steps held as fields, in the order they run.

    :attr:`STEP_FIELDS` names them, so a subclass declares its stages once and
    gets :attr:`steps` from that. An optional stage that is switched off is
    ``None`` and simply does not run.
    """

    #: The pipeline step fields, in run order.
    STEP_FIELDS: ClassVar[tuple[str, ...]] = ()

    @property
    def steps(self) -> list[StepSpec]:
        """The pipeline stages that will run, in order."""
        present = (getattr(self, name) for name in self.STEP_FIELDS)
        return [step for step in present if step is not None]


class VideoPipeline(_StepPipeline):
    """The stages run over a video input.

    Both video types use this same set of stages, but as independent blocks, so
    e.g. a room camera can be tracked with a different detector than the glasses
    cameras.
    """

    STEP_FIELDS: ClassVar[tuple[str, ...]] = (
        "object_tracking",
        "face_detection",
        "body_pose",
    )

    object_tracking: ObjectTrackingStep = Field(default_factory=ObjectTrackingStep)
    face_detection: FaceDetectionStep | None = None
    body_pose: BodyPoseStep | None = None


class SpeechPipeline(_StepPipeline):
    """The stages run over all inputs that contain audio."""

    STEP_FIELDS: ClassVar[tuple[str, ...]] = ("diarization", "transcription")

    diarization: DiarizationStep = Field(default_factory=DiarizationStep)
    transcription: TranscriptionStep | None = None


class Pipeline(_Model):
    """What to run for each type of input."""

    glasses_video: VideoPipeline = Field(default_factory=VideoPipeline)
    fixed_video: VideoPipeline = Field(default_factory=VideoPipeline)
    speech: SpeechPipeline | None = Field(default_factory=SpeechPipeline)


class ExperimentConfig(_Model):
    """The serialisable definition of an experiment: its inputs and the pipeline to run."""

    version: int = CURRENT_VERSION
    glasses_videos: list[GlassesVideoInput] = Field(default_factory=list)
    fixed_videos: list[FixedVideoInput] = Field(default_factory=list)
    audio: list[AudioInput] = Field(default_factory=list)
    pipeline: Pipeline = Field(default_factory=Pipeline)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        ids = (
            [video.id for video in self.glasses_videos]
            + [video.id for video in self.fixed_videos]
            + [audio.id for audio in self.audio]
        )
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate input ids: {sorted(duplicates)}")

        glasses_ids = {i.id for i in self.glasses_videos}
        unknown = {
            a.glasses_video
            for a in self.audio
            if a.glasses_video is not None and a.glasses_video not in glasses_ids
        }
        if unknown:
            raise ValueError(f"unknown glasses video ids: {sorted(unknown)}")
        return self
