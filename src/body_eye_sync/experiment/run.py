"""Run an :class:`~body_eye_sync.experiment.config.Experiment` non-interactively.

Drives each input through its configured pipeline stages and writes one wide
Parquet file per input (``<output_dir>/<input id>.parquet``), stamped with the
run's provenance. Machine-specific options (``device``, ONNX ``providers``) are
passed in here rather than stored in the experiment, so the same experiment file
runs anywhere.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.body_pose import detect_body_poses
from body_eye_sync.pipeline.face_detection import detect_faces
from body_eye_sync.pipeline.object_tracking import BoundingBox, detect_tracklets

logger = logging.getLogger(__name__)

#: Parquet schema-metadata key under which the run provenance JSON is stored.
PROVENANCE_KEY = "body_eye_sync.provenance"

#: Packages whose versions are recorded in the provenance for reproducibility.
_TRACKED_PACKAGES = (
    "body-eye-sync",
    "boxmot",
    "ultralytics",
    "insightface",
    "numpy",
    "pandas",
)


def run_experiment(
    experiment: Experiment,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    providers: list[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Run every input's pipeline and write one Parquet per input.

    ``output_dir`` defaults to an ``outputs`` directory beside the experiment
    file (or the current directory when the experiment was built in memory).
    Existing outputs are left untouched unless ``force`` is set. Returns a
    mapping of input id to the Parquet path written (or found).
    """
    out_dir = Path(output_dir) if output_dir is not None else experiment.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for spec in experiment.inputs:
        destination = out_dir / f"{spec.id}.parquet"
        if destination.exists() and not force:
            logger.info("skipping input %r: %s already exists", spec.id, destination)
            results[spec.id] = destination
            continue

        logger.info("running input %r", spec.id)
        video = run_input(experiment, spec, device=device, providers=providers)
        metadata = {PROVENANCE_KEY: json.dumps(_provenance(experiment, spec, device))}
        video.to_parquet(destination, metadata=metadata)
        logger.info("wrote %s", destination)
        results[spec.id] = destination
    return results


def run_input(
    experiment: Experiment,
    spec: VideoInput,
    *,
    device: str | None = None,
    providers: list[str] | None = None,
) -> Video:
    """Run one input's pipeline stages in order, returning the populated Video."""
    video_path = experiment.resolved_input_path(spec)
    if not video_path.exists():
        raise FileNotFoundError(f"input {spec.id!r} video not found: {video_path}")

    video = Video()
    video.set_video(video_path)

    # Object tracking is the required base pass; the optional passes run over
    # the tracked boxes it produces, so they always come after it.
    _run_object_tracking(video, video_path, experiment.object_tracking, device=device)

    if experiment.face_detection is not None or experiment.body_pose is not None:
        # The tracked boxes are fixed once tracking is done (the later passes only
        # add columns), so build the per-frame index once and share it.
        boxes_by_frame = video.all_boxes_by_frame()
        if experiment.face_detection is not None:
            _run_face_detection(
                video,
                video_path,
                experiment.face_detection,
                boxes_by_frame,
                providers=providers,
            )
        if experiment.body_pose is not None:
            _run_body_pose(
                video, video_path, experiment.body_pose, boxes_by_frame, device=device
            )
    return video


def _run_object_tracking(
    video: Video, video_path: Path, step: ObjectTrackingStep, *, device: str | None
) -> None:
    video.begin_object_tracking()
    for frame in detect_tracklets(
        video_path,
        detector=step.detector,
        reid=step.reid,
        tracker=step.tracker,
        object_classes=step.object_classes,
        device=device,
    ):
        video.add_object_tracking_frame(frame)
    video.finish_object_tracking()


def _run_face_detection(
    video: Video,
    video_path: Path,
    step: FaceDetectionStep,
    boxes_by_frame: dict[int, list[BoundingBox]],
    *,
    providers: list[str] | None,
) -> None:
    video.begin_face_detection()
    for result in detect_faces(
        video_path,
        boxes_by_frame,
        model_name=step.model_name,
        det_size=step.det_size,
        det_thresh=step.det_thresh,
        providers=providers,
    ):
        video.add_face_detection_frame(result)
    video.finish_face_detection()


def _run_body_pose(
    video: Video,
    video_path: Path,
    step: BodyPoseStep,
    boxes_by_frame: dict[int, list[BoundingBox]],
    *,
    device: str | None,
) -> None:
    video.begin_body_pose_detection()
    for result in detect_body_poses(
        video_path,
        boxes_by_frame,
        model_name=step.model_name,
        conf=step.conf,
        device=device,
    ):
        video.add_body_pose_frame(result)
    video.finish_body_pose_detection()


def _provenance(experiment: Experiment, spec: VideoInput, device: str | None) -> dict:
    """Machine- and run-specific facts describing how an output was produced."""
    return {
        "experiment": experiment.name,
        "input_id": spec.id,
        "video": str(experiment.resolved_input_path(spec)),
        "device": device,
        "pipeline": experiment.model_dump(
            mode="json", include=set(Experiment.STEP_FIELDS)
        ),
        "versions": _package_versions(),
        "created": datetime.now(timezone.utc).isoformat(),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def read_provenance(path: str | Path) -> dict | None:
    """Read the run provenance stamped into a Parquet output, if present."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(str(path)).metadata or {}
    raw = metadata.get(PROVENANCE_KEY.encode())
    return json.loads(raw) if raw is not None else None
