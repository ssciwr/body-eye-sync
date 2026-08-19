"""Measure and fit timeline corrections for a collection of media inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from body_eye_sync.experiment.timeline import (
    Shift,
    Timeline,
    sum_missing_before,
    to_experiment_time,
)
from body_eye_sync.preprocessing.audio import SAMPLE_RATE, audio_samples

DEFAULT_MIN_SHIFT = 0.05

SPECTRAL_HOP = 160 / SAMPLE_RATE
SPECTRAL_COEFFICIENTS = 26
SPECTRAL_MIN_QUALITY = 7.0


def spectral_features(media_path: str | Path) -> np.ndarray:
    """Cepstral features of one recording, several values per frame.

    The log-mel spectrogram comes from the faster-whisper FeatureExtractor.
    """
    from faster_whisper.feature_extractor import FeatureExtractor
    from scipy.fft import dct

    samples = audio_samples(media_path, SAMPLE_RATE)
    if len(samples) == 0:
        return np.zeros((0, SPECTRAL_COEFFICIENTS), dtype=np.float32)
    mel = np.asarray(FeatureExtractor()(samples, padding=False)).T
    cepstra = dct(mel, axis=1, norm="ortho")[:, :SPECTRAL_COEFFICIENTS]
    return (cepstra - cepstra.mean(axis=0)) / (cepstra.std(axis=0) + 1e-9)


def pairwise_offset(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Seconds to add to ``b``'s clock to reach ``a``'s, and the lock quality.

    Each spectral feature is mean-subtracted and correlated independently, then
    the correlations are combined by the length of their vector.

    Quality is how many standard deviations the best lag stands above the rest
    of the curve, so it says whether one lag is singled out rather than how
    strongly the recordings resemble each other.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("feature arrays must be two-dimensional with matching columns")
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    squares = np.zeros(size)
    for column in range(a.shape[1]):
        band = np.fft.irfft(
            np.fft.rfft(a[:, column], size) * np.conj(np.fft.rfft(b[:, column], size)),
            size,
        )
        squares += band**2
    correlation = np.sqrt(squares)
    # rearrange from FFT order into lags running -(len(b)-1) .. len(a)-1.
    correlation = np.concatenate((correlation[-(len(b) - 1) :], correlation[: len(a)]))
    lags = np.arange(-(len(b) - 1), len(a))
    peak = int(np.argmax(correlation))
    spread = correlation.std()
    quality = (
        float((correlation[peak] - correlation.mean()) / spread) if spread else 0.0
    )
    return float(lags[peak] * SPECTRAL_HOP), quality


def media_duration(path: str | Path) -> float | None:
    """How long a recording runs, read from its container without decoding it."""
    import av

    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            durations = [
                float(stream.duration * stream.time_base)
                for stream in container.streams
                if stream.duration is not None and stream.time_base is not None
            ]
            return max(durations, default=None)
    except Exception:
        return None


@dataclass
class OffsetPoint:
    time: float
    offset: float


@dataclass
class FittedTimeline:
    timeline: Timeline
    residual: float = 0.0


def offset_curve(
    reference: np.ndarray,
    other: np.ndarray,
    offset: float,
    *,
    window: float = 10.0,
    search: float = 12.0,
    min_quality: float = SPECTRAL_MIN_QUALITY,
    progress: Callable[[float], bool] | None = None,
) -> list[OffsetPoint]:
    """Measure ``other``'s offset repeatedly across the experiment.

    ``reference`` and ``other`` are arrays returned by :func:`spectral_features`.

    ``offset`` seeds the search and only has to be close enough that the true
    lag falls within ``search`` of it.

    ``window`` sets how precisely a loss can be placed

    ``min_quality`` is the lock threshold for each window.
    """
    points: list[OffsetPoint] = []
    if len(reference) == 0 or len(other) == 0:
        return points
    span = max(len(reference), len(other) + int(offset / SPECTRAL_HOP)) * SPECTRAL_HOP
    starts = np.arange(0.0, span, window / 2)
    for index, start in enumerate(starts):
        a0, a1 = int(start / SPECTRAL_HOP), int((start + window) / SPECTRAL_HOP)
        b0 = int((start - offset - search) / SPECTRAL_HOP)
        b1 = int((start + window - offset + search) / SPECTRAL_HOP)
        if progress is not None and progress((index + 1) / len(starts)) is False:
            return points
        if min(a0, b0) < 0 or a1 > len(reference) or b1 > len(other):
            continue
        lag, quality = pairwise_offset(reference[a0:a1], other[b0:b1])
        if quality < min_quality:
            continue
        points.append(OffsetPoint(start + window / 2, (a0 - b0) * SPECTRAL_HOP + lag))
    return points


def detect_shifts(
    points: list[OffsetPoint],
    *,
    min_shift: float = DEFAULT_MIN_SHIFT,
) -> list[Shift]:
    """Turn stable changes in an offset curve into missing-content shifts."""
    if len(points) < 2:
        return []
    level = None
    shifts: list[Shift] = []
    for index in range(len(points) - 1):
        before, after = points[index : index + 2]
        if abs(after.offset - before.offset) >= min_shift / 2:
            continue
        next_level = (before.offset + after.offset) / 2
        if level is None:
            level = next_level
            continue
        amount = next_level - level
        if amount < min_shift:
            continue
        shifts.append(Shift(at=before.time - level, seconds=amount))
        level = next_level
    return shifts


def fit_timeline(
    points: list[OffsetPoint],
    *,
    min_shift: float = DEFAULT_MIN_SHIFT,
) -> FittedTimeline:
    """Fit an initial offset and discrete missing content."""
    if not points:
        return FittedTimeline(Timeline())
    shifts = detect_shifts(
        points,
        min_shift=min_shift,
    )
    return fit_offset(points, shifts)


def fit_offset(points: list[OffsetPoint], shifts: list[Shift]) -> FittedTimeline:
    """Where a recording starts, given the content it lost.

    The offset is the median over the points so that a handful of badly measured
    windows cannot drag it, and the residual is how far the fitted timeline
    still misses them by.
    """
    base_offsets = [
        point.offset - sum_missing_before(shifts, point.time - point.offset)
        for point in points
    ]
    offset = float(np.median(base_offsets))
    errors = [
        to_experiment_time(point.time - point.offset, offset, shifts) - point.time
        for point in points
    ]
    residual = float(np.sqrt(np.mean(np.square(errors))))
    return FittedTimeline(Timeline(offset=offset, shifts=shifts), residual)


def missing_before_all(shifts: list[Shift], local_times: np.ndarray) -> np.ndarray:
    """:func:`sum_missing_before` for a whole array of local times at once."""
    local_times = np.asarray(local_times, dtype=float)
    missing = np.zeros(local_times.shape, dtype=float)
    for shift in shifts:
        missing[local_times >= shift.at] += shift.seconds
    return missing


def to_experiment_times(
    local_times: np.ndarray,
    offset: float,
    shifts: list[Shift],
) -> np.ndarray:
    """:func:`to_experiment_time` for a whole array of local times at once.

    For plotting a timeline, or converting a column of per-frame times, where
    calling the scalar form once per value would be the slow way to do it.
    """
    local_times = np.asarray(local_times, dtype=float)
    return local_times + offset + missing_before_all(shifts, local_times)


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
    points: dict[str, list[OffsetPoint]]
    fits: dict[str, FittedTimeline]
    # Inputs that came back without a correction
    unavailable: list[str]


def _continue(progress: Callable[[float], bool] | None, value: float) -> None:
    if progress is not None and progress(value) is False:
        raise TimingCorrectionCancelled


def _reference_with_most_overlap(features, offsets: dict[str, float]) -> str:
    """Choose the input sharing the most experiment time with all the others."""
    durations = {name: len(values) * SPECTRAL_HOP for name, values in features.items()}

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
    points: list[OffsetPoint],
    fit: FittedTimeline,
    min_total_gap: float,
) -> FittedTimeline:
    """Remove corrections too small to distinguish reliably from noise."""
    shifts = fit.timeline.shifts
    if sum(abs(shift.seconds) for shift in shifts) < min_total_gap:
        shifts = []
    if shifts == fit.timeline.shifts:
        return fit
    return fit_offset(points, shifts)


def analyse_timing_corrections(
    paths: dict[str, str | Path],
    offsets: dict[str, float],
    *,
    window: float = DEFAULT_WINDOW,
    search: float = DEFAULT_SEARCH,
    min_quality: float = SPECTRAL_MIN_QUALITY,
    min_shift: float = DEFAULT_MIN_SHIFT,
    min_total_gap: float = MIN_TOTAL_GAP,
    progress: Callable[[float], bool] | None = None,
) -> TimingCorrectionAnalysis:
    """Measure local lags and propose corrections for every usable input.

    The reference is selected automatically as the recording with the greatest
    total overlap with the others on the existing alignment timeline. Its
    clock remains unchanged; all returned fits are expressed on the existing
    experiment clock and can therefore be applied directly to the inputs.

    The complete gap correction is suppressed when its total is smaller than
    ``min_total_gap``.
    """
    if set(paths) != set(offsets):
        raise ValueError("paths and offsets must describe the same inputs")
    if len(paths) < 2:
        raise ValueError("timing correction needs at least two inputs")

    features = {}
    unavailable = []
    for index, (name, path) in enumerate(paths.items()):
        values = spectral_features(path)
        if len(values):
            features[name] = values
        else:
            unavailable.append(name)
        _continue(progress, 0.55 * (index + 1) / len(paths))
    if len(features) < 2:
        raise ValueError("fewer than two inputs have usable audio")

    reference = _reference_with_most_overlap(features, offsets)
    reference_offset = offsets[reference]
    points: dict[str, list[OffsetPoint]] = {reference: []}
    fits = {
        reference: FittedTimeline(Timeline(offset=reference_offset)),
    }
    others = [name for name in features if name != reference]
    for index, name in enumerate(others):
        start = 0.55 + 0.45 * index / len(others)
        extent = 0.45 / len(others)

        def curve_progress(value: float, start=start, extent=extent) -> bool:
            if progress is None:
                return True
            return progress(start + extent * value)

        measured = offset_curve(
            features[reference],
            features[name],
            offsets[name] - reference_offset,
            window=window,
            search=search,
            min_quality=min_quality,
            progress=curve_progress,
        )
        _continue(progress, start + extent)
        if not measured:
            unavailable.append(name)
            continue
        relative_fit = _with_sensitivity_floor(
            measured,
            fit_timeline(measured, min_shift=min_shift),
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
            relative_fit,
            timeline=replace(
                relative_fit.timeline,
                offset=relative_fit.timeline.offset + reference_offset,
            ),
        )

    _continue(progress, 1.0)
    return TimingCorrectionAnalysis(reference, points, fits, unavailable)
