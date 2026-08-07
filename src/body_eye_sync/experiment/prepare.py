"""Put an :class:`~body_eye_sync.experiment.experiment.Experiment` on one clock.

The counterpart of :mod:`body_eye_sync.experiment.run`: where that reads the
experiment and writes results, this reads the recordings and writes the
experiment's own timeline -- each input's offset, clock scale and lost content.
The pipeline reads that timeline, so this comes first.
"""

from __future__ import annotations

from typing import Callable

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.preprocessing.alignment import Alignment, TimelineFit, align_media
from body_eye_sync.preprocessing.timing_correction import TimingCorrectionAnalysis

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


def has_timing_corrections(experiment: Experiment) -> bool:
    """Whether any input carries a clock scale or lost content to clear."""
    return any(
        data.time_scale != 1.0 or data.time_shifts
        for data in recordings(experiment).values()
    )


def clear_timing_corrections(experiment: Experiment) -> list[str]:
    """Drop every input's drift and gaps, returning the ids changed.

    The offsets are left as they are: those say where each recording starts,
    which alignment worked out, and are not this correction's to undo.
    """
    cleared = []
    for name, data in recordings(experiment).items():
        if data.time_scale == 1.0 and not data.time_shifts:
            continue
        data.time_scale = 1.0
        data.time_shifts = []
        cleared.append(name)
    return cleared


def stored_fits(experiment: Experiment) -> dict[str, TimelineFit]:
    """The corrections the inputs already carry, as the fits that would give them."""
    return {
        name: TimelineFit(data.time_offset, data.time_scale, data.time_shifts)
        for name, data in recordings(experiment).items()
    }
