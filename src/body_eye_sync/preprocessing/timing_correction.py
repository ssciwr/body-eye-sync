"""Measure and fit timeline corrections for a collection of media inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from body_eye_sync.preprocessing.alignment import (
    DEFAULT_MIN_SHIFT,
    SPECTRAL,
    DriftPoint,
    TimelineFit,
    drift_curve,
    fit_offset,
    fit_timeline,
)

# The smallest clock-rate difference worth correcting
MIN_DRIFT_PPM = 30.0

# The smallest total of missing content worth correcting
MIN_TOTAL_GAP = 0.1

#: How long each measurement window is
DEFAULT_WINDOW = 10.0

#: How far either side of the existing alignment each window looks for its lag
DEFAULT_SEARCH = 12.0


class TimingCorrectionCancelled(Exception):
    """Raised when a caller cancels timing-correction analysis."""


@dataclass
class TimingCorrectionAnalysis:
    """Measured lag curves and their proposed timeline corrections."""

    reference: str
    points: dict[str, list[DriftPoint]]
    fits: dict[str, TimelineFit]
    # Inputs that came back without a correction
    unavailable: list[str]


def _continue(progress: Callable[[float], bool] | None, value: float) -> None:
    if progress is not None and progress(value) is False:
        raise TimingCorrectionCancelled


def _reference_with_most_overlap(features, offsets: dict[str, float]) -> str:
    """Choose the input sharing the most experiment time with all the others."""
    durations = {name: len(values) * SPECTRAL.hop for name, values in features.items()}

    def score(name: str) -> tuple[float, float]:
        start = offsets[name]
        end = start + durations[name]
        overlap = 0.0
        for other in features:
            if other == name:
                continue
            other_start = offsets[other]
            other_end = other_start + durations[other]
            overlap += max(0.0, min(end, other_end) - max(start, other_start))
        return overlap, durations[name]

    return max(features, key=score)


def _with_sensitivity_floor(
    points: list[DriftPoint],
    fit: TimelineFit,
    min_drift_ppm: float,
    min_total_gap: float,
) -> TimelineFit:
    """Remove corrections too small to distinguish reliably from noise."""
    scale = fit.scale
    if abs((scale - 1.0) * 1e6) < min_drift_ppm:
        scale = 1.0
    shifts = fit.shifts
    if sum(abs(shift.seconds) for shift in shifts) < min_total_gap:
        shifts = []
    if scale == fit.scale and shifts == fit.shifts:
        return fit
    # Dropping a correction moves where the recording starts, so the offset and
    # residual are placed again against the same observations.
    return fit_offset(points, scale, shifts)


def analyse_timing_corrections(
    paths: dict[str, str | Path],
    offsets: dict[str, float],
    *,
    window: float = DEFAULT_WINDOW,
    step: float | None = None,
    search: float = DEFAULT_SEARCH,
    min_quality: float | None = None,
    min_shift: float = DEFAULT_MIN_SHIFT,
    fit_drift: bool = False,
    min_drift_ppm: float = MIN_DRIFT_PPM,
    min_total_gap: float = MIN_TOTAL_GAP,
    progress: Callable[[float], bool] | None = None,
) -> TimingCorrectionAnalysis:
    """Measure local lags and propose corrections for every usable input.

    The reference is selected automatically as the recording with the greatest
    total overlap with the others on the existing alignment timeline. Its
    clock remains unchanged; all returned fits are expressed on the existing
    experiment clock and can therefore be applied directly to the inputs.

    Drift smaller than ``min_drift_ppm`` is suppressed. The complete gap
    correction is suppressed when its total is smaller than ``min_total_gap``.

    ``step`` and ``min_quality`` are left to :func:`drift_curve`, which spaces
    the windows at half a window and takes the lock threshold from the feature
    set unless either is given here.

    ``fit_drift`` decides whether a recording is allowed a clock rate of its
    own. It is off by default: a device that stalls is far commoner than one
    whose crystal is out, and a free rate will absorb a run of small losses
    into a slope that no oscillator could actually produce. Turn it on for a
    recording whose offset really does climb steadily between its steps.
    """
    if set(paths) != set(offsets):
        raise ValueError("paths and offsets must describe the same inputs")
    if len(paths) < 2:
        raise ValueError("timing correction needs at least two inputs")

    features = {}
    unavailable = []
    for index, (name, path) in enumerate(paths.items()):
        values = SPECTRAL.extract(path)
        if len(values):
            features[name] = values
        else:
            unavailable.append(name)
        _continue(progress, 0.55 * (index + 1) / len(paths))
    if len(features) < 2:
        raise ValueError("fewer than two inputs have usable audio")

    reference = _reference_with_most_overlap(features, offsets)
    reference_offset = offsets[reference]
    points: dict[str, list[DriftPoint]] = {reference: []}
    fits = {
        reference: TimelineFit(offset=reference_offset, scale=1.0),
    }
    others = [name for name in features if name != reference]
    for index, name in enumerate(others):
        start = 0.55 + 0.45 * index / len(others)
        extent = 0.45 / len(others)

        def curve_progress(value: float, start=start, extent=extent) -> bool:
            if progress is None:
                return True
            return progress(start + extent * value)

        measured = drift_curve(
            features[reference],
            features[name],
            offsets[name] - reference_offset,
            window=window,
            step=step,
            search=search,
            min_quality=min_quality,
            method=SPECTRAL,
            progress=curve_progress,
        )
        _continue(progress, start + extent)
        if not measured:
            unavailable.append(name)
            continue
        relative_fit = _with_sensitivity_floor(
            measured,
            fit_timeline(
                measured,
                min_shift=min_shift,
                time_scale=None if fit_drift else 1.0,
            ),
            min_drift_ppm,
            min_total_gap,
        )
        points[name] = [
            replace(
                point,
                time=point.time + reference_offset,
                offset=point.offset + reference_offset,
            )
            for point in measured
        ]
        fits[name] = replace(
            relative_fit, offset=relative_fit.offset + reference_offset
        )

    _continue(progress, 1.0)
    return TimingCorrectionAnalysis(reference, points, fits, unavailable)
