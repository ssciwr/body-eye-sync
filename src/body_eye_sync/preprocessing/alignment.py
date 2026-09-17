"""Calculate the time offsets that put every input on one shared timeline, based on doi.org/10.1007/s12193-015-0196-1"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from body_eye_sync.preprocessing.audio import audio_samples

LANDMARK_SAMPLE_RATE = 8000
LANDMARK_FFT = 512
LANDMARK_HOP_SAMPLES = 128
LANDMARK_HOP = LANDMARK_HOP_SAMPLES / LANDMARK_SAMPLE_RATE
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

    Quality is the number of hash matches agreeing within four landmark frames.
    Hashes occurring very often are discarded because they describe repetitive
    tones rather than distinctive acoustic events.
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
    """Offsets putting connected inputs on one clock."""

    # seconds to add to each input's own clock to reach experiment time.
    offsets: dict[str, float]
    # Requested inputs that were not able to be aligned.
    unaligned: list[str] = field(default_factory=list)


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


def align(
    envelopes: dict[str, np.ndarray],
    *,
    hop: float = LANDMARK_HOP,
    pairwise: Callable[
        [np.ndarray, np.ndarray, float], tuple[float, float]
    ] = landmark_offset,
) -> Alignment:
    """Align inputs using a maximum-quality spanning tree of pair measurements.

    Start with the input having the greatest sum of pair qualities, then
    repeatedly attach the unaligned input with the highest-quality pair to any
    already aligned input.
    """
    ids = list(envelopes)
    if not ids:
        return Alignment(offsets={})
    matched: list[PairOffset] = []
    for pair in measure_pairs(envelopes, hop, pairwise):
        if pair.quality <= 0:
            logger.warning(
                "inputs %r and %r did not match (quality %.1f); "
                "they may not overlap in time",
                pair.a,
                pair.b,
                pair.quality,
            )
            continue
        matched.append(pair)

    quality = dict.fromkeys(ids, 0.0)
    for pair in matched:
        quality[pair.a] += pair.quality
        quality[pair.b] += pair.quality
    reference = max(ids, key=quality.__getitem__)

    offsets = {reference: 0.0}
    ranked = sorted(matched, key=lambda pair: pair.quality, reverse=True)
    while len(offsets) < len(ids):
        crossing = next(
            (pair for pair in ranked if (pair.a in offsets) != (pair.b in offsets)),
            None,
        )
        if crossing is None:
            break
        if crossing.a in offsets:
            offsets[crossing.b] = offsets[crossing.a] + crossing.lag
        else:
            offsets[crossing.a] = offsets[crossing.b] - crossing.lag

    return Alignment(
        offsets=offsets, unaligned=[name for name in ids if name not in offsets]
    )


def align_media(
    paths: dict[str, str | Path],
    *,
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
    result = align(features)
    result.unaligned.extend(name for name in missing if name not in result.unaligned)
    if progress is not None:
        progress(1.0)
    return result
