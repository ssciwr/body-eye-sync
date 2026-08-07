"""Run an :class:`~body_eye_sync.experiment.experiment.Experiment` non-interactively."""

from __future__ import annotations

import logging
from pathlib import Path

from body_eye_sync.experiment.attribution import attribute_experiment_speech
from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import (
    BodyPoseStep,
    FaceDetectionStep,
    ObjectTrackingStep,
    SpeechPipeline,
    TranscriptionStep,
    VideoPipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo, Video
from body_eye_sync.pipeline.body_pose import detect_body_poses
from body_eye_sync.media import has_audio_stream
from body_eye_sync.pipeline.face_detection import detect_faces
from body_eye_sync.pipeline.object_tracking import BoundingBox, detect_tracklets
from body_eye_sync.pipeline.transcription import transcribe

logger = logging.getLogger(__name__)


def run_experiment(experiment: Experiment, *, force: bool = False) -> dict[str, Path]:
    """Run the whole pipeline over all inputs, returning each one's output directory."""
    runs = [
        *((video, run_glasses_video) for video in experiment.glasses_videos),
        *((video, run_fixed_video) for video in experiment.fixed_videos),
        *((audio, run_audio) for audio in experiment.audio),
    ]
    results: dict[str, Path] = {}
    for data, run in runs:
        directory = experiment.output_dir_for(data)
        if data.has_results(directory) and not force:
            logger.info("skipping input %r: %s already has results", data.id, directory)
            results[data.id] = directory
            continue

        logger.info("running input %r", data.id)
        run(experiment, data)
        if not data.has_data():
            # Nothing to write: an audio input with the speech pipeline off.
            logger.info("input %r produced no results", data.id)
            continue
        data.save(directory)
        logger.info("wrote %s", directory)
        results[data.id] = directory

    # Every input has been transcribed by now, so the recordings can finally be
    # related to each other: who was speaking is the one thing no single
    # recording can answer.
    _attribute_speech(experiment)
    return results


def _attribute_speech(experiment: Experiment) -> None:
    """Work out the experiment's speech turns and write them beside the inputs."""
    if experiment.pipeline.speech is None:
        return
    turns = attribute_experiment_speech(experiment)
    if not turns.has_data():
        return
    turns.save(experiment.output_dir)
    logger.info("wrote %s", experiment.output_dir / "speech_turns.parquet")


def run_glasses_video(experiment: Experiment, video: GlassesVideo) -> None:
    """Run the glasses video pipeline stages over ``video``."""
    _run_video_pipeline(video, experiment.pipeline.glasses_video)
    _run_video_speech(video, experiment.pipeline.speech)


def run_fixed_video(experiment: Experiment, video: FixedVideo) -> None:
    """Run the fixed video pipeline stages over ``video``."""
    _run_video_pipeline(video, experiment.pipeline.fixed_video)
    _run_video_speech(video, experiment.pipeline.speech)


def run_audio(experiment: Experiment, audio: Audio) -> None:
    """Run the speech pipeline stages over ``audio``."""
    if experiment.pipeline.speech is None:
        return
    audio_path = audio.audio_path
    if audio_path is None or not audio_path.exists():
        raise FileNotFoundError(f"input {audio.id!r} audio not found: {audio_path}")
    _run_speech_pipeline(audio.speech, audio_path, experiment.pipeline.speech)


def _run_video_speech(video: Video, pipeline: SpeechPipeline | None) -> None:
    """Run the speech stages over a video's audio track, if it has one."""
    if pipeline is None or video.video_path is None:
        return
    if not has_audio_stream(video.video_path):
        logger.info("input %r has no audio track; skipping speech", video.id)
        return
    _run_speech_pipeline(video.speech, video.video_path, pipeline)


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


def _run_speech_pipeline(
    speech: Speech, media_path: Path, pipeline: SpeechPipeline
) -> None:
    """Run ``pipeline``'s stages over one recording's audio, in order."""
    _run_transcription(speech, media_path, pipeline.transcription)


def _run_transcription(
    speech: Speech, media_path: Path, step: TranscriptionStep
) -> None:
    speech.begin_transcription()
    for segment in transcribe(
        media_path,
        model_name=step.model_name,
        language=step.language,
        beam_size=step.beam_size,
        device=step.device,
        vad_filter=step.vad_filter,
    ):
        speech.add_transcription_segment(segment)
    speech.finish_transcription()
