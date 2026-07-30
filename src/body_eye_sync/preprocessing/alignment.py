"""Recover the time offsets that put every input on one clock.

Recordings made on separate devices each start whenever that device was
switched on, so nothing relates them until an offset is known. Blind alignment
uses landmark fingerprints: distinctive pairs of spectrogram peaks that
survive different gains, microphones and foreground mixtures. Equal hashes
vote for a lag, so a lock means several independent acoustic events agree; it
does not become more likely merely because a long recording offers more lags
at which an ordinary correlation could peak by chance.

Offsets are solved for jointly. With N inputs there are N(N-1)/2 pair
measurements but only N-1 unknowns, so the system is overdetermined and the
least-squares residual reports whether the pairs agree with each other -- the
cheapest evidence that an alignment is real rather than a spurious peak.

Independent device clocks can also run at slightly different rates, while some
devices lose blocks of samples outright. :func:`fit_timeline` keeps those
failure modes separate: an affine clock scale describes continuous drift and
positive steps describe content that was never recorded.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

import numpy as np

from body_eye_sync.media import (
    SAMPLE_RATE,
    has_audio_stream,
    load_audio,
)

logger = logging.getLogger(__name__)

#: Seconds per envelope frame. 20 ms is far finer than the alignment a body-worn
#: microphone can support anyway: participants sit metres apart, so direct-path
#: differences of several ms come and go as they move.
ENVELOPE_HOP = 0.02

#: Seconds per spectral frame, fixed by the log-mel extractor we borrow.
FEATURE_HOP = 160 / SAMPLE_RATE

#: How many cepstral coefficients to keep. The low ones describe which formants
#: are excited -- what was said, which every microphone in the room hears alike.
#: The high ones describe fine detail that is particular to one microphone, and
#: contribute to the correlation at every lag without helping the peak, so
#: dropping them roughly doubles how far a real lock stands out.
SPECTRAL_COEFFICIENTS = 26

#: Settings for the landmark fingerprint used for blind, whole-recording
#: alignment. These follow the coarse stage of Six & Leman (2015): sparse
#: time-frequency peaks are paired and hashed, and equal hashes vote for a lag.
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

#: How far above the rest of the correlation curve a peak must stand before the
#: measurement is believed. The two outcomes separate sharply rather than
#: shading into each other: measured against recordings whose true offset is
#: known, windows above the cliff landed within 80 ms while nearly every one
#: below was wrong by whole seconds. Each feature set has its own cliff, so the
#: value belongs to the :class:`Method` rather than being global; this is the
#: loudness-envelope figure, kept for callers that pass features in directly.
MIN_QUALITY = 5.0


@dataclass
class PairOffset:
    """One measurement: how far ``b`` sits from ``a``, and how sure we are."""

    a: str
    b: str
    #: Seconds to add to ``b``'s clock to reach ``a``'s.
    lag: float
    quality: float


@dataclass
class DriftPoint:
    """One measurement of an input's offset partway through the experiment."""

    #: Experiment time at the centre of the window this was measured over.
    time: float
    #: The offset measured there, on the same footing as :attr:`Alignment.offsets`.
    offset: float
    quality: float


@dataclass
class Shift:
    """A stretch of a recording that was never written.

    ``at`` is a time on the recording's own clock, and ``seconds`` is how much
    content is missing there. Everything after it therefore sits that much
    earlier on its own clock than it does in the room, so the amount is added
    to the offset from that point on.
    """

    at: float
    seconds: float


@dataclass
class TimelineFit:
    """A clock-rate and dropped-content model fitted to local lag observations."""

    offset: float
    scale: float
    shifts: list[Shift] = field(default_factory=list)
    residual: float = 0.0

    @property
    def corrects_timing(self) -> bool:
        """Whether this says anything beyond where the recording starts.

        An offset alone is ordinary: every device was switched on at its own
        moment. A scale or a shift means the recording did not keep time.
        """
        return self.scale != 1.0 or bool(self.shifts)


@dataclass
class Alignment:
    """Offsets putting every input on one clock, and how much to trust them."""

    #: Seconds to add to each input's own clock to reach experiment time.
    offsets: dict[str, float]
    #: Requested inputs that have no locked path to the reference input.
    unaligned: list[str] = field(default_factory=list)
    #: RMS disagreement between the pair measurements and the solved offsets.
    #: A good alignment sits at a fraction of :data:`ENVELOPE_HOP`; a large value
    #: means the pairs contradict each other, so at least one is wrong.
    residual: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether every input is connected and the locked pairs agree."""
        return not self.unaligned and self.residual < ENVELOPE_HOP


def _samples(media_path: str | Path, sample_rate: int) -> np.ndarray:
    """Decoded audio for a recording, empty when there is none to read.

    A camera that recorded silently, or a file that cannot be read, is an input
    that cannot be aligned this way rather than an error, so every extractor
    treats it as having no audio rather than raising.
    """
    if not has_audio_stream(media_path):
        return np.zeros(0, dtype=np.float32)
    return np.asarray(load_audio(media_path, sample_rate), dtype=np.float32)


def spectral_features(
    media_path: str | Path,
    coefficients: int = SPECTRAL_COEFFICIENTS,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Cepstral features of one recording, several values per frame.

    Rather than only how loud a recording is, this records the shape of its
    spectrum -- roughly, what was being said. Two
    microphones in one room hear that alike even when one is far louder than
    the other, so it survives differences in level that flatten an envelope.

    The log-mel spectrogram comes from the extractor faster-whisper already
    ships, so this needs nothing that transcription did not already require.
    """
    from faster_whisper.feature_extractor import FeatureExtractor
    from scipy.fft import dct

    samples = _samples(media_path, sample_rate)
    if len(samples) == 0:
        return np.zeros((0, coefficients), dtype=np.float32)
    mel = np.asarray(FeatureExtractor()(samples, padding=False)).T
    cepstra = dct(mel, axis=1, norm="ortho")[:, :coefficients]
    return (cepstra - cepstra.mean(axis=0)) / (cepstra.std(axis=0) + 1e-9)


def landmark_features(
    media_path: str | Path,
    sample_rate: int = LANDMARK_SAMPLE_RATE,
) -> np.ndarray:
    """Sparse ``(frame, hash)`` fingerprints for blind alignment.

    Spectrogram peaks survive microphone gain, equalisation, compression and a
    considerable amount of unrelated foreground sound. Nearby peak pairs form
    high-specificity hashes. A real match therefore produces many hashes that
    all imply the same lag, whereas accidental hash collisions scatter over the
    entire recording. This is the coarse audio-alignment method described by
    Six & Leman, *Journal on Multimodal User Interfaces* 9 (2015).

    Extraction is chunked so memory use stays bounded for multi-hour media.
    """
    from scipy.ndimage import maximum_filter

    samples = _samples(media_path, sample_rate)
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
            # Do not manufacture stable fingerprints from digital silence.
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
    """Where each fingerprint occurs, keyed by hash.

    Built from ``tolist()`` rather than by indexing the array row by row, which
    for a few hundred thousand fingerprints is several times the cost of the
    whole index: it hands back plain ints instead of boxing a numpy scalar per
    value.
    """
    index: dict[int, list[int]] = {}
    for time, fingerprint in features.tolist():
        index.setdefault(fingerprint, []).append(time)
    return index


def landmark_offset(
    a: np.ndarray, b: np.ndarray, hop: float = LANDMARK_HOP
) -> tuple[float, float]:
    """Lag and confidence from agreeing landmark hashes.

    Confidence is the number of independent hash matches agreeing within four
    landmark frames, rather than an extreme-value score over every possible
    lag. Hashes occurring very often are discarded because they describe
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
    # The widest run of sorted lags spanning no more than the tolerance. For
    # each possible end, the earliest lag still within tolerance is where its
    # run begins, so one vectorised search replaces a sliding-window scan.
    starts = np.searchsorted(ordered, ordered - _LANDMARK_LAG_TOLERANCE, side="left")
    best_hi = int(np.argmax(np.arange(len(ordered)) - starts))
    inliers = order[starts[best_hi] : best_hi + 1]

    # With long recordings, a small sampling-rate mismatch spreads otherwise
    # correct lag votes into a shallow diagonal. A locally agreeing group gives
    # a safe seed for a robust affine fit, which then gathers matches from the
    # whole recording and returns the intercept (the start-time offset).
    if len(inliers) >= LANDMARK_MIN_VOTES and np.ptp(matched[inliers, 0]) * hop >= 60.0:
        for _ in range(3):
            scale, intercept = np.polyfit(matched[inliers, 0], matched[inliers, 1], 1)
            if not 0.995 <= scale <= 1.005:
                break
            errors = np.abs(matched[:, 1] - (scale * matched[:, 0] + intercept))
            expanded = np.flatnonzero(errors <= _LANDMARK_LAG_TOLERANCE)
            if len(expanded) <= len(inliers):
                break
            inliers = expanded
        if 0.995 <= scale <= 1.005:
            return float(intercept * hop), float(len(inliers))

    agreeing = lags[inliers]
    return float(np.median(agreeing) * hop), float(len(agreeing))


def pairwise_offset(
    a: np.ndarray, b: np.ndarray, hop: float = ENVELOPE_HOP
) -> tuple[float, float]:
    """Seconds to add to ``b``'s clock to reach ``a``'s, and the lock quality.

    Takes either a single feature per frame -- a loudness envelope -- or several,
    as :func:`spectral_features` produces. Features are mean-subtracted per
    dimension so that a stretch where both recordings are merely loud does not
    outscore the point where their patterns line up, and several dimensions are
    combined the way multi-band correlation usually is, by the length of the
    vector of their separate correlations.

    Quality is how many standard deviations the best lag stands above the rest
    of the curve, so it says whether one lag is singled out rather than how
    strongly the recordings resemble each other.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    if a.ndim == 1:
        correlation = np.fft.irfft(
            np.fft.rfft(a - a.mean(), size) * np.conj(np.fft.rfft(b - b.mean(), size)),
            size,
        )
    else:
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        squares = np.zeros(size)
        for column in range(a.shape[1]):
            band = np.fft.irfft(
                np.fft.rfft(a[:, column], size)
                * np.conj(np.fft.rfft(b[:, column], size)),
                size,
            )
            squares += band**2
        correlation = np.sqrt(squares)
    # Rearrange from FFT order into lags running -(len(b)-1) .. len(a)-1.
    correlation = np.concatenate((correlation[-(len(b) - 1) :], correlation[: len(a)]))
    lags = np.arange(-(len(b) - 1), len(a))
    peak = int(np.argmax(correlation))
    spread = correlation.std()
    quality = (
        float((correlation[peak] - correlation.mean()) / spread) if spread else 0.0
    )
    return float(lags[peak] * hop), quality


def measure_pairs(
    envelopes: dict[str, np.ndarray],
    hop: float = ENVELOPE_HOP,
    pairwise: Callable[
        [np.ndarray, np.ndarray, float], tuple[float, float]
    ] = pairwise_offset,
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
    min_quality: float = MIN_QUALITY,
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
    hop: float = ENVELOPE_HOP,
    min_quality: float = MIN_QUALITY,
    reference: str | None = None,
    pairwise: Callable[
        [np.ndarray, np.ndarray, float], tuple[float, float]
    ] = pairwise_offset,
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
    method: Method | None = None,
    reference: str | None = None,
    progress: Callable[[float], bool] | None = None,
) -> Alignment:
    """Offsets for a set of recordings, keyed the way the inputs are.

    Every path is read once, which is what this costs: the correlation itself
    is negligible beside it, so the work is one step per recording.

    ``progress`` is called with a fraction in ``[0, 1]`` as the reading goes on;
    returning ``False`` from it gives up, and an empty :class:`Alignment` comes
    back rather than a partial one, since offsets solved from only some of the
    recordings would not be the offsets asked for.
    """
    method = method or LANDMARK
    features = {}
    for index, (name, path) in enumerate(paths.items()):
        values = method.extract(path)
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
    result = align(
        features,
        hop=method.hop,
        min_quality=method.min_quality,
        reference=reference,
        pairwise=method.pairwise,
    )
    result.unaligned.extend(name for name in missing if name not in result.unaligned)
    if progress is not None:
        progress(1.0)
    return result


@dataclass(frozen=True)
class Method:
    """One way of reducing audio to something correlatable, and its settings.

    Two are provided. :data:`LANDMARK` is the robust default for blind
    whole-recording alignment, and :data:`SPECTRAL` follows the shape of the
    spectrum -- roughly what was said -- for repeated measurements over short
    windows.
    """

    name: str
    extract: Callable[[str | Path], np.ndarray]
    #: Seconds per frame of whatever :attr:`extract` returns.
    hop: float
    #: The lock threshold measured for this feature set; see :data:`MIN_QUALITY`.
    min_quality: float
    #: How two extracted feature sequences produce a lag and confidence.
    pairwise: Callable[[np.ndarray, np.ndarray, float], tuple[float, float]] = (
        pairwise_offset
    )


SPECTRAL = Method("spectral", spectral_features, FEATURE_HOP, 7.0)
LANDMARK = Method(
    "landmark",
    landmark_features,
    LANDMARK_HOP,
    LANDMARK_MIN_VOTES,
    landmark_offset,
)


def drift_curve(
    reference: np.ndarray,
    other: np.ndarray,
    offset: float,
    *,
    window: float = 10.0,
    step: float = 5.0,
    search: float = 12.0,
    method: Method | None = None,
    progress: Callable[[float], bool] | None = None,
) -> list[DriftPoint]:
    """Measure ``other``'s offset repeatedly across the experiment, not just once.

    :func:`align` produces one offset for a whole recording, which is all a
    device that keeps time needs. This asks the same question in windows, so a
    recording that loses content partway through shows up as a curve that moves
    rather than a constant.

    Defaults to :data:`SPECTRAL` features, which is the difference between this
    being possible and not: measured on real recordings, an energy envelope
    locks 60% of 60 s windows but only 4% of 5 s ones, while spectral features
    hold near 60% all the way down. Shortening the window is the only way to
    place a loss precisely, so the features that survive it decide the
    resolution.

    ``offset`` seeds the search and only has to be close enough that the true
    lag falls within ``search`` of it.

    ``progress`` is called with a fraction in ``[0, 1]`` as the windows are
    measured; returning ``False`` from it gives up and returns the points found
    so far, which unlike an offset are still worth having -- a curve covering
    part of the experiment still says whether that part held its time.

    ``window`` sets how precisely a loss can be placed, since a window
    straddling one reports something between the offsets either side of it.
    The default is short for that reason: a loss placed to the nearest minute
    would leave a minute of the recording carrying the wrong offset. Windows
    down to about 5 s still lock roughly half the time and still recover the
    right total, so there is little reason to use a long one.
    """
    # Only the frame rate and lock threshold are taken from the method: a drift
    # window is always correlated, since the landmark vote needs far more of a
    # recording than one window holds.
    method = method or SPECTRAL
    hop, min_quality = method.hop, method.min_quality
    points: list[DriftPoint] = []
    if len(reference) == 0 or len(other) == 0:
        return points
    span = max(len(reference), len(other) + int(offset / hop)) * hop
    starts = np.arange(0.0, span, step)
    for index, start in enumerate(starts):
        a0, a1 = int(start / hop), int((start + window) / hop)
        b0 = int((start - offset - search) / hop)
        b1 = int((start + window - offset + search) / hop)
        if progress is not None and progress((index + 1) / len(starts)) is False:
            return points
        if min(a0, b0) < 0 or a1 > len(reference) or b1 > len(other):
            continue
        lag, quality = pairwise_offset(reference[a0:a1], other[b0:b1], hop)
        if quality < min_quality:
            continue
        points.append(DriftPoint(start + window / 2, (a0 - b0) * hop + lag, quality))
    return points


def estimate_time_scale(points: list[DriftPoint]) -> float:
    """Experiment seconds per second of the input's clock.

    Each drift point identifies the same moment on the two clocks: experiment time
    is ``point.time`` and local time is ``point.time - point.offset``. The
    accumulated slope between consecutive observations estimates their
    relative rate. Intervals containing an abrupt offset change are left out,
    so dropped blocks are not mistaken for continuous clock drift. Accumulating
    before dividing is important: ordinary oscillator drift moves by much less
    than one feature frame between adjacent observations and is therefore seen
    as an occasional one-frame step rather than a smooth per-window change.
    """
    if len(points) < 2:
        return 1.0
    ordered = sorted(points, key=lambda point: point.time)
    experiment = np.asarray([point.time for point in ordered])
    offsets = np.asarray([point.offset for point in ordered])
    experiment_delta = np.diff(experiment)
    offset_delta = np.diff(offsets)
    local_delta = experiment_delta - offset_delta
    valid = (experiment_delta > 0) & (local_delta > 0)
    if not np.any(valid):
        return 1.0
    # Do not let a long stretch without a lock dominate one finite difference.
    usual_gap = np.median(experiment_delta[valid])
    valid &= experiment_delta <= 3 * usual_gap
    # A real dropped block changes lag abruptly; clock-rate quantisation and
    # ordinary measurement noise stay below the smallest supported drop.
    valid &= np.abs(offset_delta) < 0.08
    # A short correlation window measures a drop as a two- or three-point ramp.
    # Exclude all intervals in a neighbourhood whose cumulative change is drop
    # sized even when no individual increment crosses the threshold.
    for start in range(len(offset_delta)):
        end = min(start + 3, len(offset_delta))
        if abs(offsets[end] - offsets[start]) >= 0.06:
            valid[start:end] = False
    if not np.any(valid):
        return 1.0
    return float(experiment_delta[valid].sum() / local_delta[valid].sum())


def _remove_clock_drift(points: list[DriftPoint], scale: float) -> list[DriftPoint]:
    """Offsets after accounting for the continuous difference in clock rate."""
    return [
        DriftPoint(
            point.time,
            point.time - scale * (point.time - point.offset),
            point.quality,
        )
        for point in points
    ]


def _split_cost(cumulative, squares, lo, hi):
    """Sum of squared deviations from the mean over ``[lo, hi)``."""
    span = hi - lo
    if span <= 0:
        return 0.0
    total = cumulative[hi] - cumulative[lo]
    return (squares[hi] - squares[lo]) - total * total / span


def _change_points(offsets, lo, hi, plateau, min_shift, found):
    """Split a stretch wherever its level genuinely changes, and recurse.

    Chooses the split that best explains the stretch as two levels rather than
    one, which is what makes this work on a curve measured in short windows: a
    window straddling a loss matches no single lag and reports something in
    between, so the step arrives as a ramp a few windows wide and never appears
    between one window and the next. Comparing whole stretches sees it;
    comparing neighbours does not.
    """
    if hi - lo < 2 * plateau:
        return
    cumulative = np.concatenate(([0.0], np.cumsum(offsets[lo:hi])))
    squares = np.concatenate(([0.0], np.cumsum(offsets[lo:hi] ** 2)))
    whole = _split_cost(cumulative, squares, 0, hi - lo)
    best, best_gain = None, 0.0
    for split in range(plateau, hi - lo - plateau + 1):
        gain = whole - (
            _split_cost(cumulative, squares, 0, split)
            + _split_cost(cumulative, squares, split, hi - lo)
        )
        if gain > best_gain:
            best, best_gain = split, gain
    if best is None:
        return
    split = lo + best
    step = np.median(offsets[split:hi]) - np.median(offsets[lo:split])
    if abs(step) < min_shift:
        return
    found.append(split)
    _change_points(offsets, lo, split, plateau, min_shift, found)
    _change_points(offsets, split, hi, plateau, min_shift, found)


def detect_shifts(
    points: list[DriftPoint],
    *,
    min_shift: float = 0.08,
    plateau: int = 2,
    merge_within: float | None = None,
    time_scale: float | None = None,
) -> list[Shift]:
    """Turn a drift curve into the list of losses that would explain it.

    The curve is treated as flat stretches separated by steps, because that is
    what dropped content looks like: the offset holds while the recording keeps
    up and jumps when a piece of it is missing. Each step becomes a
    :class:`Shift` on the recording's own clock. Continuous clock-rate drift is
    estimated and removed before those steps are sought.

    ``min_shift`` is the smallest step worth calling a loss. Measured against
    recordings known not to drift, noise reaches 60 ms and produced spurious
    losses below 0.06; measured against ones that do, real losses start being
    swallowed above 0.10. The default sits between.

    ``plateau`` is how many windows a level needs before it is believed, and
    losses closer together than ``merge_within`` are reported as one. That
    matters because a loss is measured as a ramp a few windows wide rather than
    a clean step, and a ramp can otherwise be split into several smaller losses
    that are really one. Left unset it is twice the spacing of the windows,
    which is about as close as two losses can be told apart anyway.
    """
    if len(points) < 2 * plateau:
        return []
    time_scale = estimate_time_scale(points) if time_scale is None else time_scale
    points = _remove_clock_drift(points, time_scale)
    times = np.array([p.time for p in points])
    offsets = np.array([p.offset for p in points])

    splits: list[int] = []
    _change_points(offsets, 0, len(offsets), plateau, min_shift, splits)
    bounds = [0, *sorted(splits), len(offsets)]

    levels = [
        (times[lo], float(np.median(offsets[lo:hi])))
        for lo, hi in zip(bounds[:-1], bounds[1:])
        if hi > lo
    ]
    if not levels:
        return []

    # Gather the changes on the experiment clock. Merging has to happen here
    # rather than after converting to the recording's own clock, because each
    # conversion subtracts a different accumulated offset and so distorts how
    # far apart two changes appear to be.
    changes = [
        (start, after - before)
        for (_, before), (start, after) in zip(levels[:-1], levels[1:])
    ]
    if merge_within is None:
        spacing = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        merge_within = 2 * spacing
    # Each change is compared with the last one taken into the group rather
    # than the group's start, so a ramp -- which is a run of small steps, each
    # a window apart -- collects into the single loss it represents.
    merged: list[list[float]] = []
    for when, amount in changes:
        if merged and when - merged[-1][2] <= merge_within:
            merged[-1][1] += amount
            merged[-1][2] = when
        else:
            merged.append([when, amount, when])

    shifts: list[Shift] = []
    running = levels[0][1]
    for when, amount, _ in merged:
        if amount >= min_shift:
            # The window centres run from half a window either side of the
            # loss, so the level changes over the loss itself; its own clock
            # reading is that experiment time less the offset before it.
            shifts.append(Shift(at=(when - running) / time_scale, seconds=amount))
        running += amount
    return shifts


def fit_timeline(
    points: list[DriftPoint],
    *,
    min_shift: float = 0.08,
    plateau: int = 2,
    merge_within: float | None = None,
    time_scale: float | None = None,
) -> TimelineFit:
    """Fit continuous clock drift and discrete missing content together.

    Continuous sampling-rate error becomes :attr:`TimelineFit.scale`; positive
    steps left after removing that slope become missing-content shifts. This
    keeps the two physically different failure modes separate. Pass a fixed
    ``time_scale`` (most usefully ``1.0``) to constrain the fit instead of
    estimating a continuous rate difference.
    """
    if not points:
        return TimelineFit(offset=0.0, scale=1.0)
    scale = estimate_time_scale(points) if time_scale is None else time_scale
    shifts = detect_shifts(
        points,
        min_shift=min_shift,
        plateau=plateau,
        merge_within=merge_within,
        time_scale=scale,
    )
    return fit_offset(points, scale, shifts)


def fit_offset(
    points: list[DriftPoint], scale: float, shifts: list[Shift]
) -> TimelineFit:
    """Where a recording starts, given a clock rate and the content it lost.

    ``scale`` and ``shifts`` fix the shape of the timeline; this places it
    against the observations. The offset is the median over the points so that
    a handful of badly measured windows cannot drag it, and the residual is how
    far the fitted timeline still misses them by.
    """
    base_offsets = [
        point.time
        - scale * (point.time - point.offset)
        - missing_before(shifts, point.time - point.offset)
        for point in points
    ]
    offset = float(np.median(base_offsets))
    errors = [
        to_experiment_time(point.time - point.offset, offset, shifts, scale)
        - point.time
        for point in points
    ]
    residual = float(np.sqrt(np.mean(np.square(errors))))
    return TimelineFit(offset=offset, scale=scale, shifts=shifts, residual=residual)


def missing_before(shifts: list[Shift], local_time: float) -> float:
    """How much content is missing before a moment on a recording's own clock.

    The amount every shift up to ``local_time`` lost, which is what separates
    the recording's own clock from the experiment's beyond a plain offset. Order
    does not matter to a total, so the shifts need not be sorted.
    """
    return sum(shift.seconds for shift in shifts if shift.at <= local_time)


def to_experiment_time(
    local_time: float,
    offset: float,
    shifts: list[Shift],
    scale: float = 1.0,
) -> float:
    """Experiment time for a moment on one recording's own clock.

    ``offset`` is where the recording starts, ``scale`` is experiment seconds
    per local second, and each :class:`Shift` before ``local_time`` adds the
    content missing at that point. Always defined: every moment the recording
    holds did happen.
    """
    return scale * local_time + offset + missing_before(shifts, local_time)


def missing_before_all(shifts: list[Shift], local_times: np.ndarray) -> np.ndarray:
    """:func:`missing_before` for a whole array of local times at once."""
    local_times = np.asarray(local_times, dtype=float)
    missing = np.zeros(local_times.shape, dtype=float)
    for shift in shifts:
        missing[local_times >= shift.at] += shift.seconds
    return missing


def to_experiment_times(
    local_times: np.ndarray,
    offset: float,
    shifts: list[Shift],
    scale: float = 1.0,
) -> np.ndarray:
    """:func:`to_experiment_time` for a whole array of local times at once.

    For plotting a timeline, or converting a column of per-frame times, where
    calling the scalar form once per value would be the slow way to do it.
    """
    local_times = np.asarray(local_times, dtype=float)
    return scale * local_times + offset + missing_before_all(shifts, local_times)


def to_local_time(
    experiment_time: float,
    offset: float,
    shifts: list[Shift],
    scale: float = 1.0,
) -> float | None:
    """Where a moment of the experiment sits on one recording's own clock.

    ``None`` when the recording holds no such moment, which is what a
    :class:`Shift` means: the content was never written, so a stretch of the
    experiment has nowhere to map to. Callers that would otherwise silently claim
    a recording covered something it missed need that answer.
    """
    running = offset
    for shift in sorted(shifts, key=lambda value: value.at):
        if experiment_time < scale * shift.at + running:
            return (experiment_time - running) / scale
        running += shift.seconds
        if experiment_time < scale * shift.at + running:
            return None  # inside the stretch that was never recorded
    return (experiment_time - running) / scale


def unobserved(
    offset: float, shifts: list[Shift], scale: float = 1.0
) -> list[tuple[float, float]]:
    """The stretches of experiment time a recording has nothing for.

    One per :class:`Shift`, as ``(start, end)`` on the experiment clock. An
    analysis working in intervals should split on these or mark them, rather
    than reading straight through a stretch that was never recorded.
    """
    spans: list[tuple[float, float]] = []
    running = offset
    for shift in sorted(shifts, key=lambda value: value.at):
        start = scale * shift.at + running
        running += shift.seconds
        spans.append((start, scale * shift.at + running))
    return spans
