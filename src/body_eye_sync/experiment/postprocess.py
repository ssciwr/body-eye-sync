"""Postprocess an experiment using the pipeline outputs."""

from __future__ import annotations

import logging

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.postprocessing.attribution import (
    Progress,
    attribute_segments,
    measure_levels,
)

logger = logging.getLogger(__name__)


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
