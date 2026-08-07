"""Work out who spoke when across a whole experiment.

The counterpart of :mod:`body_eye_sync.experiment.run`: where that runs models
over one input at a time, this relates the inputs to each other. It reads the
transcripts those per-input passes produced and the recordings' loudness, and
writes the experiment's speech turns.

Only the glasses recordings can be attributed, because only they belong to a
known wearer. A fixed camera or a standalone microphone has nobody behind it, so
its transcript says what was said but never who said it.
"""

from __future__ import annotations

import logging
from typing import Callable

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.speech_turns import SpeechTurns
from body_eye_sync.experiment.video import GlassesVideo
from body_eye_sync.preprocessing.alignment import TimelineFit
from body_eye_sync.preprocessing.attribution import (
    BLEED_AGREEMENT,
    LIVE_ABOVE_FLOOR_DB,
    OWNERSHIP_SHARE,
    Levels,
    attribute_segments,
    measure_levels,
)

logger = logging.getLogger(__name__)

Progress = Callable[[float], bool]


def wearers(experiment: Experiment) -> dict[str, GlassesVideo]:
    """The glasses recordings that could be attributed, keyed by input id.

    A glasses video needs both its recording and a transcript of it: the
    recording says how loud its wearer was, the transcript says what was said.
    """
    return {
        video.id: video
        for video in experiment.glasses_videos
        if video.path is not None and video.speech.data is not None
    }


def measure_experiment_levels(
    experiment: Experiment, *, progress: Progress | None = None
) -> Levels:
    """Measure how loud each wearer's microphone is, on the experiment clock."""
    inputs = wearers(experiment)
    return measure_levels(
        {name: data.path for name, data in inputs.items()},
        _fits(inputs),
        progress=progress,
    )


def attribute_experiment_speech(
    experiment: Experiment,
    *,
    threshold: float = LIVE_ABOVE_FLOOR_DB,
    ownership: float = OWNERSHIP_SHARE,
    agreement: float = BLEED_AGREEMENT,
    progress: Progress | None = None,
) -> SpeechTurns:
    """Work out the experiment's speech turns and store them on it.

    Returns the turns, which are also left on ``experiment`` so they are saved
    with it. An experiment with fewer than two transcribed glasses recordings
    has nothing to compare, and comes back with no turns at all.
    """
    inputs = wearers(experiment)
    if len(inputs) < 2:
        logger.info(
            "cannot attribute speech: %d transcribed glasses recording(s), need 2",
            len(inputs),
        )
        experiment.speech_turns.clear()
        return experiment.speech_turns

    levels = measure_experiment_levels(experiment, progress=progress)
    turns = attribute_segments(
        {name: data.speech.data for name, data in inputs.items()},
        levels,
        _fits(inputs),
        words={name: data.speech.words for name, data in inputs.items()},
        threshold=threshold,
        ownership=ownership,
        agreement=agreement,
    )
    experiment.speech_turns.set_data(turns)
    if progress is not None:
        progress(1.0)
    logger.info(
        "attributed %d speech turns across %d wearers",
        len(turns),
        turns["speaker"].nunique() if not turns.empty else 0,
    )
    return experiment.speech_turns


def _fits(inputs: dict[str, GlassesVideo]) -> dict[str, TimelineFit]:
    """Each input's timeline, as the fit that would produce it."""
    return {
        name: TimelineFit(data.time_offset, data.time_scale, data.time_shifts)
        for name, data in inputs.items()
    }
