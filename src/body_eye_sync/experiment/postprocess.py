"""Postprocess an experiment using the pipeline outputs."""

from __future__ import annotations

import logging

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.postprocessing.attribution import (
    Progress,
    attribute_segments,
    measure_levels,
)
from body_eye_sync.postprocessing.tracklets_clustering import cluster_tracklets

logger = logging.getLogger(__name__)


def clustering_blocked_reason(experiment: Experiment) -> str | None:
    """Why the experiment is not ready for tracklet clustering, if anything."""
    if not experiment.glasses_videos:
        return "Add glasses videos first, in the Input files tab."
    for video in experiment.glasses_videos:
        if video.data is None or "face_score" not in video.data.columns:
            return f"Run tracking and face detection for {video.id!r} first."
        if video.data["face_score"].notna().any() and video.face_embeddings is None:
            return f"Collect face recognition embeddings for {video.id!r} first."
    return None


def cluster_experiment_tracklets(
    experiment: Experiment,
    *,
    debug: bool = False,
) -> None:
    """Cluster videos, infer glasses wearers, and store tracklet identities.

    Only glasses videos contribute clustering evidence and receive entries in
    ``experiment.identities``. Fixed videos are ignored. Face
    detection must have completed for every glasses video before inferring
    wearers; an unprocessed recording cannot supply evidence of absence.

    Visibility counts are independent of timeline offsets/rates; gaze processing
    uses the shared clock.
    """
    settings = experiment.pipeline.cluster_post_processing
    experiment.identities.set_data(
        cluster_tracklets(
            experiment.glasses_videos,
            **settings.model_dump(),
            debug=debug,
        )
    )


def attribute_experiment_speech(
    experiment: Experiment,
    *,
    progress: Progress | None = None,
) -> None:
    """Work out the experiment's speech turns and store them on it."""
    settings = experiment.pipeline.speech_post_processing

    inputs = {
        video.id: video
        for video in experiment.glasses_videos
        if video.path is not None and video.speech.data is not None
    }
    if len(inputs) < 2:
        logger.info(
            "cannot attribute speech: %d transcribed glasses recording(s), need 2",
            len(inputs),
        )
        experiment.speech_turns.clear()
        return

    timelines = {name: data.timeline for name, data in inputs.items()}
    levels = measure_levels(
        {name: data.loudness.data for name, data in inputs.items()},
        timelines,
        settings,
    )
    turns = attribute_segments(
        {name: data.speech.data for name, data in inputs.items()},
        levels,
        timelines,
        settings,
        words={name: data.speech.words for name, data in inputs.items()},
        progress=progress,
    )
    experiment.speech_turns.set_data(turns)
    if progress is not None:
        progress(1.0)
    logger.info(
        "attributed %d speech turns across %d wearers",
        len(turns),
        turns["speaker"].nunique() if not turns.empty else 0,
    )
