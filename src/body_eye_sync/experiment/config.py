"""Serialisable definition of an experiment: its inputs and the pipeline to run."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The on-disk format version. Only needs to be bumped for non-backward-compatible changes, e.g. a new required field.
CURRENT_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class GlassesVideoInput(_Input):
    """Video recorded by a participant's glasses-mounted camera."""


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


InputSpec = Union[GlassesVideoInput, FixedVideoInput, AudioInput]


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
            "Upper bound on the number of speakers, or -1 to use the threshold "
            "instead. Not an exact count: fewer speakers may be found, and the "
            "default of -1 is usually more accurate. Prefer tuning threshold."
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


class VideoPipeline(_Model):
    """The stages run over a video input.

    Both video types use this same set of stages, but as independent blocks, so
    e.g. a room camera can be tracked with a different detector than the glasses
    cameras.
    """

    # The pipeline step fields, in run order
    STEP_FIELDS: ClassVar[tuple[str, ...]] = (
        "object_tracking",
        "face_detection",
        "body_pose",
    )

    object_tracking: ObjectTrackingStep = Field(default_factory=ObjectTrackingStep)
    face_detection: FaceDetectionStep | None = None
    body_pose: BodyPoseStep | None = None

    @property
    def steps(self) -> list[StepSpec]:
        """The pipeline stages that will run, in order."""
        present = (getattr(self, name) for name in self.STEP_FIELDS)
        return [step for step in present if step is not None]


class AudioPipeline(_Model):
    """The stages run over an audio input.

    Diarization always runs: it produces the speech turns that are the rows of
    the output, so transcription has nothing to attribute its text to without
    it. Transcription is optional, for experiments that only need to know who
    spoke when.
    """

    # The pipeline step fields, in run order
    STEP_FIELDS: ClassVar[tuple[str, ...]] = ("diarization", "transcription")

    diarization: DiarizationStep = Field(default_factory=DiarizationStep)
    transcription: TranscriptionStep | None = None

    @property
    def steps(self) -> list[StepSpec]:
        """The pipeline stages that will run, in order."""
        present = (getattr(self, name) for name in self.STEP_FIELDS)
        return [step for step in present if step is not None]


class Pipeline(_Model):
    """What to run for each type of input."""

    glasses_video: VideoPipeline = Field(default_factory=VideoPipeline)
    fixed_video: VideoPipeline = Field(default_factory=VideoPipeline)
    audio: AudioPipeline = Field(default_factory=AudioPipeline)


class ExperimentConfig(_Model):
    """The serialisable definition of an experiment: its inputs and the pipeline to run."""

    version: int = CURRENT_VERSION
    name: str
    glasses_videos: list[GlassesVideoInput] = Field(default_factory=list)
    fixed_videos: list[FixedVideoInput] = Field(default_factory=list)
    audio: list[AudioInput] = Field(default_factory=list)
    pipeline: Pipeline = Field(default_factory=Pipeline)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if not self.inputs:
            raise ValueError("experiment has no inputs")

        ids = [i.id for i in self.inputs]
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

    @property
    def inputs(self) -> list[InputSpec]:
        """Every input of every type, for the checks that span all of them.

        Anything that acts on inputs of a particular type should go through that
        type's own list instead, which keeps the type known.
        """
        return [*self.glasses_videos, *self.fixed_videos, *self.audio]
