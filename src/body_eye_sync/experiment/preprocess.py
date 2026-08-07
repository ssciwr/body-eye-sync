"""Prepare an experiment's inputs for the pipeline, e.g. aligning them and applying timing corrections."""

from __future__ import annotations

from typing import Callable

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.preprocessing.alignment import Alignment, align_media
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
            inputs[name].timeline.offset = offset
    return alignment


def apply_timing_corrections(
    experiment: Experiment, analysis: TimingCorrectionAnalysis
) -> list[str]:
    """Write an analysis' corrections onto the inputs, returning the ids changed.

    Only inputs that actually need a correction are modified.
    """
    inputs = recordings(experiment)
    corrected = {
        name: fit
        for name, fit in analysis.fits.items()
        if fit.timeline.corrects_timing and name in inputs
    }
    for name, fit in corrected.items():
        data = inputs[name]
        data.timeline.offset = fit.timeline.offset
        data.timeline.shifts = list(fit.timeline.shifts)
    return list(corrected)


def has_timing_corrections(experiment: Experiment) -> bool:
    """Whether any input carries lost content to clear."""
    return any(data.timeline.shifts for data in recordings(experiment).values())


def clear_timing_corrections(experiment: Experiment) -> list[str]:
    """Drop every input's gaps, returning the ids changed.

    The offsets are left as they are: those say where each recording starts,
    which alignment worked out, and are not this correction's to undo.
    """
    cleared = []
    for name, data in recordings(experiment).items():
        if not data.timeline.shifts:
            continue
        data.timeline.shifts = []
        cleared.append(name)
    return cleared
