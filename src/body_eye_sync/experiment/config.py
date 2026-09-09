"""Serialisable definition of an experiment: its inputs and the pipeline to run."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The on-disk format version. Only needs to be bumped for non-backward-compatible changes, e.g. a new required field.
CURRENT_VERSION = 1


def validate_input_id(input_id: str) -> str:
    """Return an input id, having checked it is safe for generated names."""
    if not input_id:
        raise ValueError("input id cannot be empty")
    if any(char in input_id for char in ("/", "\\", "[", "]")) or input_id in (
        ".",
        "..",
    ):
        raise ValueError(f"input id cannot contain reserved characters: {input_id!r}")
    return input_id


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimelineConfig(_Model):
    """Where one recording sits on the experiment's shared clock."""

    offset: float = Field(
        0.0,
        description=(
            "Seconds to add to this input's own clock to place it on the shared "
            "experiment timeline."
        ),
    )
    rate: float = Field(
        1.0,
        gt=0,
        description=(
            "Experiment seconds per second of this input's own clock. Every "
            "device counts time on its own crystal, and two of them differ by "
            "tens of parts per million, which is tens of milliseconds across a "
            "long recording."
        ),
    )


class _Input(_Model):
    """Fields shared by every experiment input, whatever its type.

    ``path`` may be relative; it is resolved against the experiment folder by the
    runtime :class:`Experiment`. Each input's type is given by which list of
    :class:`ExperimentConfig` it appears in, so the specs carry no ``kind`` tag.
    """

    id: str
    path: Path
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)

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

    Embedded audio in video files is handled as part of video playback; this
    input is for separate audio files.
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


class TranscriptionStep(_Model):
    """Speech transcription. Fields mirror ``transcribe``."""

    model_name: str = Field(
        "primeline/whisper-large-v3-turbo-german",
        description=(
            "Whisper model. The primeLine models are accuracy-tuned for German; "
            "CrisperWhisper models produce verbatim transcripts; large-v3 is "
            "the strongest general multilingual choice."
        ),
        json_schema_extra={
            "choices": [
                "primeline/whisper-large-v3-turbo-german",
                "primeline/whisper-large-v3-german",
                "nyralabs/CrisperWhisper2.0_large",
                "nyralabs/CrisperWhisper2.0_medium",
                "large-v3",
                "large-v3-turbo",
                "distil-large-v3",
                "tiny",
                "base",
                "small",
                "medium",
            ]
        },
    )
    language: str | None = Field(
        "de",
        description=(
            "ISO 639-1 language code of the recording, e.g. 'de'. German is the "
            "accuracy-first default; leave unset to detect the language from the "
            "first 30 seconds."
        ),
    )
    beam_size: int = Field(5, ge=1, description="Decoding beam width.")
    vad_filter: bool = Field(
        False,
        description=(
            "Skip silent stretches, which speeds up the pass and suppresses text "
            "invented over silence."
        ),
    )


class SpeechPostProcessingSettings(_Model):
    """How transcripts are combined to form experiment-wide speaker turns."""

    split_gap_seconds: float = Field(
        0.75,
        ge=0,
        description=(
            "Split a segment when consecutive words are separated by at "
            "least this many seconds. Sentence-ending punctuation also splits."
        ),
    )
    split_on_sentence_end: bool = Field(
        True,
        description="Split after words ending in sentence punctuation (. ! ? …).",
    )
    split_on_comma: bool = Field(
        True,
        description=(
            "Split at commas when both adjacent clauses meet the minimum word "
            "count and duration below."
        ),
    )
    minimum_clause_words: int = Field(
        4,
        ge=1,
        description=("Minimum number of words required on each side of a comma split."),
    )
    minimum_clause_seconds: float = Field(
        0.5,
        ge=0,
        description=(
            "Minimum duration required on each side of a comma split, in seconds."
        ),
    )
    floor_percentile: float = Field(
        10.0,
        ge=0,
        le=100,
        description=(
            "Use this percentile of each recording's levels as its quiet floor."
        ),
    )
    live_above_floor_db: float = Field(
        10.0,
        ge=0,
        description=(
            "A recording counts as carrying speech when its level is this "
            "many dB above its own quiet floor."
        ),
    )
    ownership_share: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "Keep a piece when this recording is the loudest for more than this "
            "fraction of its active frames, or live for more than this fraction of all frames."
        ),
    )
    fuzzy_agreement: float = Field(
        0.6,
        ge=0,
        le=1,
        description=(
            "Treat overlapping text as the same voice when its "
            "normalized character similarity exceeds this fraction."
        ),
    )


# A pipeline stage for type hints
StepSpec = Union[
    ObjectTrackingStep,
    FaceDetectionStep,
    BodyPoseStep,
    TranscriptionStep,
]


class StepPipeline(_Model):
    """Base class for pipeline configurations edited by the shared GUI."""


class VideoPipeline(StepPipeline):
    """The stages run over a video input.

    Both video types use this same set of stages, but as independent blocks, so
    e.g. a room camera can be tracked with a different detector than the glasses
    cameras.
    """

    object_tracking: ObjectTrackingStep = Field(default_factory=ObjectTrackingStep)
    face_detection: FaceDetectionStep | None = None
    body_pose: BodyPoseStep | None = None


class SpeechPipeline(StepPipeline):
    """The stages run over all inputs that contain audio.

    Transcription is the only one: it says what was said, and who said it is
    settled afterwards, by comparing the experiment's recordings with each other.
    """

    transcription: TranscriptionStep = Field(default_factory=TranscriptionStep)


class Pipeline(_Model):
    """What to run for each type of input."""

    glasses_video: VideoPipeline = Field(default_factory=VideoPipeline)
    fixed_video: VideoPipeline = Field(default_factory=VideoPipeline)
    speech: SpeechPipeline | None = Field(default_factory=SpeechPipeline)
    speech_post_processing: SpeechPostProcessingSettings = Field(
        default_factory=SpeechPostProcessingSettings
    )


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
