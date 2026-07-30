"""Put an :class:`~body_eye_sync.experiment.experiment.Experiment` on one clock.

The counterpart of :mod:`body_eye_sync.experiment.run`: where that reads the
experiment and writes results, this reads the recordings and writes the
experiment's own timeline -- each input's offset, clock scale and lost content.
The pipeline reads that timeline, so this comes first.
"""

from __future__ import annotations

import logging
from typing import Callable

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.preprocessing.alignment import Alignment, TimelineFit, align_media
from body_eye_sync.preprocessing.timing_correction import (
    TimingCorrectionAnalysis,
    analyse_timing_corrections,
)

logger = logging.getLogger(__name__)

Progress = Callable[[float], bool]


def recordings(experiment: Experiment) -> dict[str, Video | Audio]:
    """The experiment's inputs that have a recording to measure, keyed by id."""
    return {data.id: data for data in experiment.inputs if data.path is not None}


def align_experiment(
    experiment: Experiment, *, progress: Progress | None = None
) -> Alignment:
    """Measure where each input starts and write the offsets onto the inputs."""
    inputs = recordings(experiment)
    if len(inputs) < 2:
        # Nothing to align against: one recording is its own timeline.
        return Alignment(offsets={})
    alignment = align_media(
        {name: data.path for name, data in inputs.items()}, progress=progress
    )
    for name, offset in alignment.offsets.items():
        if name in inputs:
            inputs[name].time_offset = offset
    return alignment


def correct_experiment_timing(
    experiment: Experiment, *, progress: Progress | None = None
) -> tuple[TimingCorrectionAnalysis | None, list[str]]:
    """Measure clock drift and lost content, returning what was corrected.

    The analysis is ``None`` when there is nothing to compare. Otherwise it
    comes back with the ids of the inputs it corrected, in analysis order.
    """
    inputs = recordings(experiment)
    if len(inputs) < 2:
        return None, []
    analysis = analyse_timing_corrections(
        {name: data.path for name, data in inputs.items()},
        {name: data.time_offset for name, data in inputs.items()},
        progress=progress,
    )
    return analysis, apply_timing_corrections(experiment, analysis)


def apply_timing_corrections(
    experiment: Experiment, analysis: TimingCorrectionAnalysis
) -> list[str]:
    """Write an analysis' corrections onto the inputs, returning the ids changed.

    Only inputs that actually need a correction are touched. A fit's offset and
    scale are a matched pair, so an input being corrected takes the offset that
    goes with its slope; one that held time keeps the offset alignment gave it,
    rather than having it refined behind the user's back.
    """
    inputs = recordings(experiment)
    corrected = {
        name: fit
        for name, fit in analysis.fits.items()
        if fit.corrects_timing and name in inputs
    }
    for name, fit in corrected.items():
        data = inputs[name]
        data.time_offset = fit.offset
        data.time_scale = fit.scale
        data.time_shifts = list(fit.shifts)
    return list(corrected)


def stored_fits(experiment: Experiment) -> dict[str, TimelineFit]:
    """The corrections the inputs already carry, as the fits that would give them."""
    return {
        name: TimelineFit(data.time_offset, data.time_scale, data.time_shifts)
        for name, data in recordings(experiment).items()
    }


def prepare_experiment(
    experiment: Experiment, *, progress: Progress | None = None
) -> TimingCorrectionAnalysis | None:
    """Align the inputs and correct their clocks, in that order.

    Timing correction measures drift against the existing alignment, so the
    offsets have to be in place before it runs.
    """
    inputs = recordings(experiment)
    if len(inputs) < 2:
        logger.info("nothing to align: the experiment has fewer than two recordings")
        return None

    alignment = align_experiment(experiment, progress=_part(progress, 0.0, 0.5))
    for name in alignment.unaligned:
        logger.warning("input %r could not be aligned to the others", name)
    logger.info("aligned %d input(s)", len(alignment.offsets))

    analysis, corrected = correct_experiment_timing(
        experiment, progress=_part(progress, 0.5, 0.5)
    )
    if analysis is not None:
        for name in analysis.unavailable:
            logger.warning("input %r could not be measured for drift or gaps", name)
        logger.info(
            "corrected drift or gaps in %d input(s)%s",
            len(corrected),
            f": {', '.join(corrected)}" if corrected else "",
        )
    return analysis


def _part(progress: Progress | None, start: float, extent: float) -> Progress | None:
    """Map a stage's own 0-1 progress onto its share of the whole run."""
    if progress is None:
        return None
    return lambda value: progress(start + extent * value)
