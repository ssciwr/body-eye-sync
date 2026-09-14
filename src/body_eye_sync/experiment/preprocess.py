"""Prepare an experiment's inputs for the pipeline, e.g. aligning them and correcting their clock rates."""

from __future__ import annotations

from typing import Callable

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.preprocessing.alignment import Alignment, align_media
from body_eye_sync.preprocessing.clock_rate import ClockRateAnalysis

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
            inputs[name].timeline.offset = offset
    return alignment


def apply_clock_rates(experiment: Experiment, analysis: ClockRateAnalysis) -> list[str]:
    """Write an analysis' findings onto the inputs, returning the ids changed.

    Significant drift fits replace the whole timeline. Successfully measured
    inputs without significant drift keep their alignment offset and return to
    a unit rate, clearing a correction an earlier analysis applied.
    """
    inputs = recordings(experiment)
    changed = []
    for name in analysis.points.keys() | analysis.fits.keys():
        if name not in inputs:
            continue
        data = inputs[name]
        fit = analysis.fits.get(name)
        offset = data.timeline.offset if fit is None else fit.offset
        rate = 1.0 if fit is None else fit.rate
        if (data.timeline.offset, data.timeline.rate) == (offset, rate):
            continue
        data.timeline.offset = offset
        data.timeline.rate = rate
        changed.append(name)
    return changed


def has_corrected_clock_rates(experiment: Experiment) -> bool:
    """Whether any input carries a clock-rate correction to clear."""
    return any(data.timeline.corrects_drift for data in recordings(experiment).values())


def clear_clock_rates(experiment: Experiment) -> list[str]:
    """Drop every input's clock-rate correction, returning the ids changed.

    The offsets are left as they are: those say where each recording starts,
    which alignment worked out, and are not this correction's to undo.
    """
    cleared = []
    for name, data in recordings(experiment).items():
        if not data.timeline.corrects_drift:
            continue
        data.timeline.rate = 1.0
        cleared.append(name)
    return cleared
