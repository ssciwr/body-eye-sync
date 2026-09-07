"""Calculate the time offsets that put every input on one shared timeline, based on doi.org/10.1007/s12193-015-0196-1"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from body_eye_sync.preprocessing.audio import audio_samples

ALIGNMENT_TOLERANCE = 0.02

LANDMARK_SAMPLE_RATE = 8000
LANDMARK_FFT = 512
LANDMARK_HOP_SAMPLES = 128
LANDMARK_HOP = LANDMARK_HOP_SAMPLES / LANDMARK_SAMPLE_RATE
LANDMARK_MIN_VOTES = 7.0
_LANDMARK_PEAK_BLOCK = 8
_LANDMARK_PEAKS_PER_BLOCK = 3
_LANDMARK_TARGETS = 5
_LANDMARK_MIN_DELTA = 2
_LANDMARK_MAX_DELTA = 64
_LANDMARK_LAG_TOLERANCE = 4
_LANDMARK_MAX_OCCURRENCES = 20

logger = logging.getLogger(__name__)


def landmark_features(
    media_path: str | Path,
    sample_rate: int = LANDMARK_SAMPLE_RATE,
) -> np.ndarray:
    """Sparse ``(frame, hash)`` fingerprints for blind alignment."""
    from scipy.ndimage import maximum_filter

    samples = audio_samples(media_path, sample_rate)
    if len(samples) < LANDMARK_FFT:
        return np.empty((0, 2), dtype=np.int64)

    frame_count = 1 + (len(samples) - LANDMARK_FFT) // LANDMARK_HOP_SAMPLES
    window = np.hanning(LANDMARK_FFT).astype(np.float32)
    peaks: list[tuple[int, int, float]] = []
    chunk_frames = 4096
    time_radius = 2
    low_bin, high_bin = 3, LANDMARK_FFT // 2 - 2

    for core_start in range(0, frame_count, chunk_frames):
        core_end = min(core_start + chunk_frames, frame_count)
        ext_start = max(0, core_start - time_radius)
        ext_end = min(frame_count, core_end + time_radius)
        starts = np.arange(ext_start, ext_end) * LANDMARK_HOP_SAMPLES
        indices = starts[:, None] + np.arange(LANDMARK_FFT)
        frames = samples[indices] * window
        magnitude = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        log_magnitude = np.log1p(1000.0 * magnitude)
        local_max = maximum_filter(log_magnitude, size=(5, 7), mode="nearest")
        candidates = log_magnitude == local_max
        candidates[:, :low_bin] = False
        candidates[:, high_bin + 1 :] = False

        first_group = core_start // _LANDMARK_PEAK_BLOCK
        last_group = (core_end + _LANDMARK_PEAK_BLOCK - 1) // _LANDMARK_PEAK_BLOCK
        for group in range(first_group, last_group):
            lo = max(group * _LANDMARK_PEAK_BLOCK, core_start)
            hi = min((group + 1) * _LANDMARK_PEAK_BLOCK, core_end)
            local_lo, local_hi = lo - ext_start, hi - ext_start
            sample_lo = lo * LANDMARK_HOP_SAMPLES
            sample_hi = min(
                len(samples),
                (hi - 1) * LANDMARK_HOP_SAMPLES + LANDMARK_FFT,
            )
            if np.sqrt(np.mean(samples[sample_lo:sample_hi] ** 2)) < 1e-5:
                continue
            where = np.argwhere(candidates[local_lo:local_hi])
            if len(where) == 0:
                continue
            values = log_magnitude[
                where[:, 0] + local_lo,
                where[:, 1],
            ]
            baseline = float(np.median(log_magnitude[local_lo:local_hi]))
            useful = np.flatnonzero(values > baseline + 1.0)
            if len(useful) == 0:
                continue
            useful = useful[np.argsort(values[useful])[-_LANDMARK_PEAKS_PER_BLOCK:]]
            for selected in useful:
                frame = int(where[selected, 0] + lo)
                frequency = int(where[selected, 1])
                peaks.append((frame, frequency, float(values[selected])))

    if len(peaks) < 2:
        return np.empty((0, 2), dtype=np.int64)
    peaks.sort()
    peak_times = np.asarray([p[0] for p in peaks])
    fingerprints: list[tuple[int, int]] = []
    for anchor_time, anchor_frequency, _ in peaks:
        lo = int(np.searchsorted(peak_times, anchor_time + _LANDMARK_MIN_DELTA))
        hi = int(np.searchsorted(peak_times, anchor_time + _LANDMARK_MAX_DELTA + 1))
        if hi <= lo:
            continue
        targets = sorted(peaks[lo:hi], key=lambda p: p[2], reverse=True)[
            :_LANDMARK_TARGETS
        ]
        for target_time, target_frequency, _ in targets:
            delta = target_time - anchor_time
            fingerprint = (anchor_frequency << 16) | (target_frequency << 8) | delta
            fingerprints.append((anchor_time, fingerprint))
    return np.asarray(fingerprints, dtype=np.int64).reshape(-1, 2)


def _hash_index(features: np.ndarray) -> dict[int, list[int]]:
    """Where each fingerprint occurs, keyed by hash."""
    index: dict[int, list[int]] = {}
    for time, fingerprint in features.tolist():
        index.setdefault(fingerprint, []).append(time)
    return index


def landmark_offset(
    a: np.ndarray, b: np.ndarray, hop: float = LANDMARK_HOP
) -> tuple[float, float]:
    """How many seconds to add to b's clock to align with a.

    Confidence is the number of independent hash matches agreeing within four
    landmark frames. Hashes occurring very often are discarded because they describe
    repetitive tones rather than distinctive acoustic events.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0
    a_by_hash = _hash_index(a)
    b_by_hash = _hash_index(b)

    matches: list[tuple[int, int]] = []
    for fingerprint in a_by_hash.keys() & b_by_hash.keys():
        a_times = a_by_hash[fingerprint]
        b_times = b_by_hash[fingerprint]
        if (
            len(a_times) > _LANDMARK_MAX_OCCURRENCES
            or len(b_times) > _LANDMARK_MAX_OCCURRENCES
        ):
            continue
        matches.extend((tb, ta) for ta in a_times for tb in b_times)
    if not matches:
        return 0.0, 0.0

    matched = np.asarray(matches, dtype=float)
    lags = matched[:, 1] - matched[:, 0]
    order = np.argsort(lags)
    ordered = lags[order]
    # find largest collection of matches with offsets that agree within four landmark frames
    starts = np.searchsorted(ordered, ordered - _LANDMARK_LAG_TOLERANCE, side="left")
    best_hi = int(np.argmax(np.arange(len(ordered)) - starts))
    inliers = order[starts[best_hi] : best_hi + 1]

    agreeing = lags[inliers]
    return float(np.median(agreeing) * hop), float(len(agreeing))


@dataclass
class PairOffset:
    """One measurement: how far ``b`` sits from ``a``, and how sure we are."""

    a: str
    b: str
    # lag is the seconds to add to ``b``'s clock to reach ``a``'s.
    lag: float
    quality: float


@dataclass
class Alignment:
    """Offsets putting every input on one clock, and how much to trust them."""

    # seconds to add to each input's own clock to reach experiment time.
    offsets: dict[str, float]
    # requested inputs that have no locked path to the reference input.
    unaligned: list[str] = field(default_factory=list)
    # RMS disagreement between the pair measurements and the solved offsets
    residual: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether every input is connected and the locked pairs agree."""
        return not self.unaligned and self.residual < ALIGNMENT_TOLERANCE


def measure_pairs(
    envelopes: dict[str, np.ndarray],
    hop: float = LANDMARK_HOP,
    pairwise: Callable[
        [np.ndarray, np.ndarray, float], tuple[float, float]
    ] = landmark_offset,
) -> list[PairOffset]:
    """Measure every pair of envelopes against each other."""
    return [
        PairOffset(a, b, *pairwise(envelopes[a], envelopes[b], hop))
        for a, b in itertools.combinations(envelopes, 2)
    ]


def solve_offsets(
    ids: list[str],
    pairs: list[PairOffset],
    *,
    min_quality: float = LANDMARK_MIN_VOTES,
    reference: str | None = None,
) -> Alignment:
    """Least-squares offsets from pair measurements, ignoring ones that failed.

    Each locked pair contributes ``offset(b) - offset(a) = lag``, weighted by
    its quality, and the reference is pinned to zero. With more pairs than
    unknowns the leftover disagreement becomes :attr:`Alignment.residual`.
    """
    if not ids:
        return Alignment(offsets={})
    reference = reference or ids[0]
    if reference not in ids:
        raise ValueError(f"reference input {reference!r} is not available")

    locked: list[PairOffset] = []
    for pair in pairs:
        if pair.quality < min_quality:
            logger.warning(
                "inputs %r and %r did not lock (quality %.1f); "
                "they may not overlap in time",
                pair.a,
                pair.b,
                pair.quality,
            )
            continue
        locked.append(pair)

    neighbours = {name: set() for name in ids}
    for pair in locked:
        neighbours[pair.a].add(pair.b)
        neighbours[pair.b].add(pair.a)
    connected = {reference}
    frontier = [reference]
    while frontier:
        name = frontier.pop()
        for neighbour in neighbours[name] - connected:
            connected.add(neighbour)
            frontier.append(neighbour)
    solved_ids = [name for name in ids if name in connected]
    unaligned = [name for name in ids if name not in connected]
    index = {name: i for i, name in enumerate(solved_ids)}

    rows: list[np.ndarray] = []
    values: list[float] = []
    weights: list[float] = []
    for pair in locked:
        if pair.a not in connected or pair.b not in connected:
            continue
        row = np.zeros(len(solved_ids))
        row[index[pair.b]] = 1.0
        row[index[pair.a]] = -1.0
        rows.append(row)
        values.append(pair.lag)
        # Weighted least squares multiplies rows by sqrt(weight).
        weights.append(np.sqrt(pair.quality))

    # Pin the reference to zero, weighted so the solve cannot trade it away.
    pin = np.zeros(len(solved_ids))
    pin[index[reference]] = 1.0
    rows.append(pin)
    values.append(0.0)
    weights.append(max(weights, default=1.0) * 100)

    design = np.asarray(rows)
    measured = np.asarray(values)
    weight = np.asarray(weights)
    solution, *_ = np.linalg.lstsq(
        design * weight[:, None], measured * weight, rcond=None
    )
    # Residual over the pair equations only; the pin is a constraint, not data.
    leftover = design[:-1] @ solution - measured[:-1]
    residual = float(np.sqrt((leftover**2).mean())) if len(leftover) else 0.0
    return Alignment(
        offsets={name: float(solution[index[name]]) for name in solved_ids},
        unaligned=unaligned,
        residual=residual,
    )


def align(
    envelopes: dict[str, np.ndarray],
    *,
    hop: float = LANDMARK_HOP,
    min_quality: float = LANDMARK_MIN_VOTES,
    reference: str | None = None,
    pairwise: Callable[
        [np.ndarray, np.ndarray, float], tuple[float, float]
    ] = landmark_offset,
) -> Alignment:
    """Solve every input's offset from the envelopes, using all pairs at once.

    ``reference`` is the input left at offset zero, defaulting to the first;
    which one is chosen only shifts the whole timeline, it does not change the
    inputs' positions relative to each other.
    """
    ids = list(envelopes)
    return solve_offsets(
        ids,
        measure_pairs(envelopes, hop, pairwise),
        min_quality=min_quality,
        reference=reference,
    )


def align_media(
    paths: dict[str, str | Path],
    *,
    reference: str | None = None,
    progress: Callable[[float], bool] | None = None,
) -> Alignment:
    """Offsets for a set of recordings, keyed the way the inputs are."""
    features = {}
    for index, (name, path) in enumerate(paths.items()):
        values = landmark_features(path)
        if len(values) == 0:
            logger.warning("input %r has no audio to align on; skipping", name)
        else:
            features[name] = values
        # Reading is nearly all of the work, so it gets nearly all of the bar.
        if progress is not None and progress(0.95 * (index + 1) / len(paths)) is False:
            return Alignment(offsets={})
    missing = [name for name in paths if name not in features]
    if reference is not None and reference not in features:
        logger.warning("reference input %r has no audio to align on", reference)
        return Alignment(offsets={}, unaligned=list(paths))
    result = align(features, reference=reference)
    result.unaligned.extend(name for name in missing if name not in result.unaligned)
    if progress is not None:
        progress(1.0)
    return result
