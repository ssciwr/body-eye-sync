"""Run an :class:`~body_eye_sync.experiment.experiment.Experiment` non-interactively."""

from __future__ import annotations

import logging
from pathlib import Path

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.body_pose import detect_body_poses
from body_eye_sync.pipeline.face_detection import detect_faces
from body_eye_sync.pipeline.object_tracking import BoundingBox, detect_tracklets

logger = logging.getLogger(__name__)


def run_experiment(experiment: Experiment, *, force: bool = False) -> dict[str, Path]:
    """Run every input's pipeline and write one Parquet per input.

    Outputs go to the experiment's ``outputs`` folder; existing ones are left
    untouched unless ``force`` is set. Returns a mapping of input id to the
    Parquet path written (or found).
    """
    results: dict[str, Path] = {}
    for spec in experiment.config.inputs:
        destination = experiment.output_path(spec)
        if destination.exists() and not force:
            logger.info("skipping input %r: %s already exists", spec.id, destination)
            results[spec.id] = destination
            continue

        logger.info("running input %r", spec.id)
        video = run_input(experiment, spec)
        destination.parent.mkdir(parents=True, exist_ok=True)
        video.to_parquet(destination)
        logger.info("wrote %s", destination)
        results[spec.id] = destination
    return results


def run_input(experiment: Experiment, spec: VideoInput) -> Video:
    """Run one input's pipeline stages in order, returning the populated Video."""
    video_path = experiment.resolved_input_path(spec)
    if not video_path.exists():
        raise FileNotFoundError(f"input {spec.id!r} video not found: {video_path}")

    config = experiment.config
    video = Video()
    video.set_video(video_path)
    _run_object_tracking(video, video_path, config.object_tracking)
    boxes_by_frame = video.all_boxes_by_frame()
    if config.face_detection is not None:
        _run_face_detection(video, video_path, config.face_detection, boxes_by_frame)
    if config.body_pose is not None:
        _run_body_pose(video, video_path, config.body_pose, boxes_by_frame)
    return video


def _run_object_tracking(
    video: Video, video_path: Path, step: ObjectTrackingStep
) -> None:
    video.begin_object_tracking(step.embeddings_per_track)
    for frame in detect_tracklets(
        video_path,
        detector=step.detector,
        reid=step.reid,
        tracker=step.tracker,
        object_classes=step.object_classes,
    ):
        video.add_object_tracking_frame(frame)
    video.finish_object_tracking()


def _run_face_detection(
    video: Video,
    video_path: Path,
    step: FaceDetectionStep,
    boxes_by_frame: dict[int, list[BoundingBox]],
) -> None:
    video.begin_face_detection(step.embeddings_per_track)
    for result in detect_faces(
        video_path,
        boxes_by_frame,
        model_name=step.model_name,
        det_size=step.det_size,
        det_thresh=step.det_thresh,
    ):
        video.add_face_detection_frame(result)
    video.finish_face_detection()


def _run_body_pose(
    video: Video,
    video_path: Path,
    step: BodyPoseStep,
    boxes_by_frame: dict[int, list[BoundingBox]],
) -> None:
    video.begin_body_pose_detection()
    for result in detect_body_poses(
        video_path,
        boxes_by_frame,
        model_name=step.model_name,
        conf=step.conf,
    ):
        video.add_body_pose_frame(result)
    video.finish_body_pose_detection()
