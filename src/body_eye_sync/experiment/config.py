"""Serialisable definition of an experiment: its inputs and the pipeline to run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The on-disk format version. Only needs to be bumped for non-backward-compatible changes, e.g. a new required field.
CURRENT_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoInput(_Model):
    """A single video file to process, referenced by a stable ``id``.

    ``path`` may be relative; it is resolved against the experiment folder by the
    runtime :class:`Experiment`.
    """

    kind: Literal["video"] = "video"
    id: str
    path: Path


# An experiment input - currently only VideoInput but e.g. AudioInput etc. can be added here in the future
InputSpec = Annotated[Union[VideoInput], Field(discriminator="kind")]


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


# A pipeline stage for type hints
StepSpec = Union[ObjectTrackingStep, FaceDetectionStep, BodyPoseStep]


class ExperimentConfig(_Model):
    """The serialisable definition of an experiment: its inputs and the pipeline to run."""

    # The pipeline step fields, in run order
    STEP_FIELDS: ClassVar[tuple[str, ...]] = (
        "object_tracking",
        "face_detection",
        "body_pose",
    )

    version: int = CURRENT_VERSION
    name: str
    inputs: list[InputSpec]
    object_tracking: ObjectTrackingStep = Field(default_factory=ObjectTrackingStep)
    face_detection: FaceDetectionStep | None = None
    body_pose: BodyPoseStep | None = None

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if not self.inputs:
            raise ValueError("experiment has no inputs")

        ids = [i.id for i in self.inputs]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate input ids: {sorted(duplicates)}")
        return self

    @property
    def steps(self) -> list[StepSpec]:
        """The pipeline stages that will run, in order."""
        present = (getattr(self, name) for name in self.STEP_FIELDS)
        return [step for step in present if step is not None]
