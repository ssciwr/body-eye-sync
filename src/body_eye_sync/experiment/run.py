"""Run an :class:`~body_eye_sync.experiment.experiment.Experiment` non-interactively."""

from __future__ import annotations

import logging
from pathlib import Path

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoPipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo, Video
from body_eye_sync.pipeline.body_pose import detect_body_poses
from body_eye_sync.pipeline.face_detection import detect_faces
from body_eye_sync.pipeline.object_tracking import BoundingBox, detect_tracklets

logger = logging.getLogger(__name__)


def run_experiment(experiment: Experiment, *, force: bool = False) -> dict[str, Path]:
    """Run the whole pipeline over all inputs"""
    runs = [
        *((video, run_glasses_video) for video in experiment.glasses_videos),
        *((video, run_fixed_video) for video in experiment.fixed_videos),
    ]
    results: dict[str, Path] = {}
    for video, run in runs:
        destination = experiment.output_path(video)
        if destination.exists() and not force:
            logger.info("skipping input %r: %s already exists", video.id, destination)
            results[video.id] = destination
            continue

        logger.info("running input %r", video.id)
        run(experiment, video)
        destination.parent.mkdir(parents=True, exist_ok=True)
        video.to_parquet(destination)
        logger.info("wrote %s", destination)
        results[video.id] = destination

    for audio in experiment.audio:
        logger.info("skipping input %r: audio has no pipeline stages yet", audio.id)
    return results


def run_glasses_video(experiment: Experiment, video: GlassesVideo) -> None:
    """Run the glasses video pipeline stages over ``video``."""
    _run_video_pipeline(video, experiment.pipeline.glasses_video)


def run_fixed_video(experiment: Experiment, video: FixedVideo) -> None:
    """Run the fixed video pipeline stages over ``video``."""
    _run_video_pipeline(video, experiment.pipeline.fixed_video)


def _run_video_pipeline(video: Video, pipeline: VideoPipeline) -> None:
    """Run ``pipeline``'s stages over ``video``, in order."""
    video_path = video.video_path
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"input {video.id!r} video not found: {video_path}")
    _run_object_tracking(video, video_path, pipeline.object_tracking)
    boxes_by_frame = video.all_boxes_by_frame()
    if pipeline.face_detection is not None:
        _run_face_detection(video, video_path, pipeline.face_detection, boxes_by_frame)
    if pipeline.body_pose is not None:
        _run_body_pose(video, video_path, pipeline.body_pose, boxes_by_frame)


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
