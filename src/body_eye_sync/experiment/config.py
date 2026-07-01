"""Serialisable definition of an experiment: its inputs and the pipeline to run.

An :class:`Experiment` captures the reproducible, machine-independent parts of a
run -- which files to process and which model steps to apply with which
arguments -- so it can be written to and reloaded from a YAML file and shared.

The schema is built to extend cleanly: new input modalities become members of
:data:`InputSpec` (tagged by a ``kind`` literal so old files keep parsing), and
new pipeline stages become new optional fields on :class:`Experiment`. Object
tracking is the required base pass; every other stage runs over its tracked
boxes and is optional (``None`` when switched off).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

# The on-disk format version. Only needs to be bumped for non-backward-compatible changes.
CURRENT_VERSION = 1

# An experiment lives in a folder; this is the config file inside it.
DEFAULT_EXPERIMENT_FILENAME = "experiment.yaml"

# Subfolder of the experiment folder which contains the output dataframes as Parquet files.
OUTPUTS_DIRNAME = "outputs"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoInput(_Model):
    """A single video file to process, referenced by a stable ``id``.

    ``path`` may be relative; it is resolved against the experiment file's
    directory by :meth:`Experiment.resolved_input_path`.
    """

    kind: Literal["video"] = "video"
    id: str
    path: Path


# An experiment input - currently only VideoInput but e.g. AudioInput etc. can be added here in the future
InputSpec = Annotated[Union[VideoInput], Field(discriminator="kind")]


class ObjectTrackingStep(_Model):
    """Object detection + ReID tracking. Fields mirror ``detect_tracklets``.

    ``choices`` in a field's ``json_schema_extra`` are suggested values the GUI
    offers in an *editable* combobox; a custom value is still allowed.

    The defaults are the product's recommended recent medium-size models -- a
    good speed/accuracy balance -- which is deliberately heavier than the
    lightweight defaults the ``detect_*`` pipeline functions fall back to.
    """

    detector: str = Field(
        "yolo26m",
        description="Object detector model (smaller = faster, larger = more accurate).",
        json_schema_extra={
            # Only the current YOLO26 generation is suggested (n<s<m<l<x). The
            # field is free-form, so any other model name can still be typed.
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
                # OSNet, increasing width/capacity
                "osnet_x0_25_msmt17",
                "osnet_x0_5_msmt17",
                "osnet_x0_75_msmt17",
                "osnet_x1_0_msmt17",
                "osnet_ain_x1_0_msmt17",
                # larger models
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
            "Number of best body-appearance (ReID) embeddings kept per tracklet "
            "(ranked by detection confidence) for later identity clustering. "
            "0 disables body embeddings."
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
        description=(
            "Number of best face embeddings kept per tracklet (ranked by face "
            "score) for later identity clustering. 0 disables face embeddings."
        ),
    )


class BodyPoseStep(_Model):
    """Per-box body-pose detection. Fields mirror ``detect_body_poses``."""

    model_name: str = Field(
        "yolo26m-pose.pt",
        description="Ultralytics YOLO pose weights (smaller = faster, larger = more accurate).",
        json_schema_extra={
            # Only the current YOLO26 generation is suggested (n<s<m<l<x). The
            # field is free-form, so any other model name can still be typed.
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


# A pipeline stage. A plain union used only for type hints; the discrete steps
# are now named fields on Experiment rather than a discriminated list.
StepSpec = Union[ObjectTrackingStep, FaceDetectionStep, BodyPoseStep]


class Experiment(_Model):
    """A complete, serialisable experiment definition.

    Round-trips to YAML via :meth:`from_yaml`/:meth:`to_yaml`. Relative input
    paths are resolved against the loaded file's directory.

    The pipeline is a fixed shape rather than a free-form list: object tracking
    is the required base pass and always runs first; :attr:`face_detection` and
    :attr:`body_pose` are optional passes over its tracked boxes (``None`` when
    switched off). New stages are added as further optional fields.
    """

    #: The pipeline step fields, in run order (object tracking first). The single
    #: source of truth for which fields make up the pipeline -- ``steps`` and the
    #: run provenance both derive from it, so a new stage is added in one place.
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

    #: Directory the experiment was loaded from; used to resolve relative paths.
    _base_dir: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check(self) -> Experiment:
        if not self.inputs:
            raise ValueError("experiment has no inputs")

        ids = [i.id for i in self.inputs]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate input ids: {sorted(duplicates)}")
        return self

    @property
    def steps(self) -> list[StepSpec]:
        """The pipeline stages that will run, in order (tracking first)."""
        present = (getattr(self, name) for name in self.STEP_FIELDS)
        return [step for step in present if step is not None]

    @classmethod
    def from_dir(cls, folder: str | Path) -> Experiment:
        """Load the experiment stored in ``folder`` (its ``experiment.yaml``)."""
        return cls.from_yaml(Path(folder) / DEFAULT_EXPERIMENT_FILENAME)

    def to_dir(self, folder: str | Path) -> Path:
        """Write this experiment into ``folder`` as ``experiment.yaml``.

        Creates ``folder`` if needed, records it as the base directory, and
        returns the path of the written config file.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / DEFAULT_EXPERIMENT_FILENAME
        self.to_yaml(path)
        self._base_dir = folder
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        """Load and validate an experiment from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        version = data.get("version", CURRENT_VERSION)
        if version > CURRENT_VERSION:
            raise ValueError(
                f"experiment version {version} is newer than supported {CURRENT_VERSION}; "
                "please upgrade body-eye-sync"
            )
        experiment = cls.model_validate(data)
        experiment._base_dir = path.parent
        return experiment

    def to_yaml(self, path: str | Path) -> None:
        """Write this experiment to a YAML file (paths kept as written)."""
        data = self.model_dump(mode="json")
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @property
    def base_dir(self) -> Path:
        """The experiment folder, or the current directory if not loaded/saved."""
        return self._base_dir if self._base_dir is not None else Path.cwd()

    @property
    def output_dir(self) -> Path:
        """Where per-input Parquet outputs are written, inside the folder."""
        return self.base_dir / OUTPUTS_DIRNAME

    def output_path(self, spec: VideoInput) -> Path:
        """Parquet output path for ``spec`` under :attr:`output_dir`."""
        return self.output_dir / f"{spec.id}.parquet"

    def resolved_input_path(self, spec: VideoInput) -> Path:
        """Absolute path for ``spec``, relative paths taken from the folder.

        When the experiment was not loaded from a file, relative paths are taken
        from the current working directory.
        """
        if spec.path.is_absolute():
            return spec.path
        return (self.base_dir / spec.path).resolve()
