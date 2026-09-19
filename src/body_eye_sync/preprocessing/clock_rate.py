"""Determine offset and rate to make recordings align in time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.preprocessing.audio import SAMPLE_RATE, audio_samples


SPECTRAL_HOP = 160 / SAMPLE_RATE
SPECTRAL_COEFFICIENTS = 26
SPECTRAL_MIN_QUALITY = 7.0

#: Below this, a measured clock difference is noise rather than a difference.
MIN_DRIFT_PPM = 2.0

#: A slope needs at least this much recording behind it to mean anything.
MIN_DRIFT_SPAN = 60.0

#: And at least this many measurements, before an interval around it means much.
MIN_DRIFT_POINTS = 8

#: How sure of a clock difference to be before correcting for it.
DRIFT_CONFIDENCE = 0.95


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


@dataclass
class OffsetPoint:
    time: float
    offset: float


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

    ``window`` is how much audio each measurement correlates. Windows do not
    overlap, so their errors are independent and the spread of the points is
    an honest measure of how well the lag is known.

    ``min_quality`` is the lock threshold for each window.
    """
    points: list[OffsetPoint] = []
    if len(reference) == 0 or len(other) == 0:
        return points
    span = max(len(reference), len(other) + int(offset / SPECTRAL_HOP)) * SPECTRAL_HOP
    starts = np.arange(0.0, span, window)
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


def fit_timeline(
    points: list[OffsetPoint], *, min_drift_ppm: float = MIN_DRIFT_PPM
) -> Timeline | None:
    """Fit where a drifting recording starts and how fast its clock runs.

    The fit is a straight line through the measured offsets, so its slope is
    the difference between the two devices' clocks. It is a Theil-Sen line —
    the median of the slopes between every pair of points — because a window
    that locks onto the wrong lag misses by a second where the others agree to
    a few milliseconds, and least squares would follow it.

    ``None`` unless the slope clears three separate bars: measured over at least
    ``MIN_DRIFT_SPAN`` of recording, so it is not extrapolated across a session
    from a moment of it; a ``DRIFT_CONFIDENCE`` interval that excludes no
    difference at all, so it is not noise; and at least ``min_drift_ppm``, so
    it is worth correcting.
    """
    if not points:
        return None
    local = np.asarray([point.time - point.offset for point in points])
    experiment = np.asarray([point.time for point in points])
    rate = _fitted_rate(local, experiment, min_drift_ppm)
    if rate is None:
        return None
    # also use the median for the offset to reduce effect of outliers
    offset = float(np.median(experiment - local * rate))
    return Timeline(offset=offset, rate=rate)


def _fitted_rate(
    local: np.ndarray, experiment: np.ndarray, min_drift_ppm: float
) -> float | None:
    """The non-unit clock rate the points support, if any."""
    from scipy.stats import theilslopes

    if len(local) < MIN_DRIFT_POINTS or float(np.ptp(local)) < MIN_DRIFT_SPAN:
        return None
    slope, _, low, high = theilslopes(experiment, local, DRIFT_CONFIDENCE)
    if low <= 1.0 <= high:
        return None
    if abs(float(slope) - 1.0) * 1e6 < min_drift_ppm:
        return None
    return float(slope)


#: How long each measurement window is
DEFAULT_WINDOW = 10.0

#: How far either side of the existing alignment each window looks for its lag
DEFAULT_SEARCH = 12.0


class ClockRateAnalysisCancelled(Exception):
    """Raised when a caller cancels clock-rate analysis."""


@dataclass
class ClockRateAnalysis:
    """Measured lag curves and the timelines fitted to them."""

    reference: str
    points: dict[str, list[OffsetPoint]]
    #: Significant non-unit clock-rate fits, keyed by input id.
    fits: dict[str, Timeline]
    #: Inputs for which no usable offset measurements could be made.
    unavailable: list[str]


def _continue(progress: Callable[[float], bool] | None, value: float) -> None:
    if progress is not None and progress(value) is False:
        raise ClockRateAnalysisCancelled


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


def analyse_clock_rates(
    paths: dict[str, str | Path],
    offsets: dict[str, float],
    *,
    window: float = DEFAULT_WINDOW,
    search: float = DEFAULT_SEARCH,
    min_quality: float = SPECTRAL_MIN_QUALITY,
    min_drift_ppm: float = MIN_DRIFT_PPM,
    progress: Callable[[float], bool] | None = None,
) -> ClockRateAnalysis:
    """Measure local lags and fit a timeline for every usable input.

    The reference is selected automatically as the recording with the greatest
    total overlap with the others on the existing alignment timeline. Its
    clock is the one the others are measured against, so it keeps a rate of
    one; all returned fits are expressed on the existing experiment clock and
    can therefore be applied directly to the inputs.

    A clock difference smaller than ``min_drift_ppm`` is left uncorrected.
    """
    if set(paths) != set(offsets):
        raise ValueError("paths and offsets must describe the same inputs")
    if len(paths) < 2:
        raise ValueError("clock-rate analysis needs at least two inputs")

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
    fits: dict[str, Timeline] = {}
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
        points[name] = [
            replace(
                point,
                time=point.time + reference_offset,
                offset=point.offset + reference_offset,
            )
            for point in measured
        ]
        relative_fit = fit_timeline(measured, min_drift_ppm=min_drift_ppm)
        if relative_fit is not None:
            fits[name] = replace(
                relative_fit,
                offset=relative_fit.offset + reference_offset,
            )

    _continue(progress, 1.0)
    return ClockRateAnalysis(reference, points, fits, unavailable)
