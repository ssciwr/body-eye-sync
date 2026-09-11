from __future__ import annotations

from itertools import chain
from pathlib import Path

import numpy as np

from body_eye_sync.media import container_origin

SAMPLE_RATE = 16000


def _origin(container, sample_rate: int, fallback: int) -> int:
    """Output sample index the container's timeline starts at."""
    origin = container_origin(container)
    if origin is None:
        return fallback
    return round(origin * sample_rate)


def load_audio(audio_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a recording onto its own timeline, as mono samples.

    Every decoded frame is placed at the position its timestamp gives it, so
    stretches a recorder never wrote are represented as silence.
    """
    import av

    with av.open(str(audio_path)) as container:
        return _decode_audio(container, sample_rate)


def _decode_audio(container, sample_rate: int) -> np.ndarray:
    import av

    if not container.streams.audio:
        return np.zeros(0, dtype=np.float32)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    pieces: list[tuple[int, np.ndarray]] = []
    # a trailing None flushes whatever the resampler is still holding.
    for frame in chain(container.decode(stream), [None]):
        for resampled in resampler.resample(frame):
            samples = resampled.to_ndarray().reshape(-1)
            if resampled.pts is not None and samples.size:
                pieces.append((resampled.pts, samples))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    origin = _origin(container, sample_rate, pieces[0][0])

    length = max(pts - origin + len(samples) for pts, samples in pieces)
    # Placed straight into the float buffer, so the int16 samples are never
    # held twice over.
    buffer = np.zeros(max(length, 0), dtype=np.float32)
    for pts, samples in pieces:
        start = pts - origin
        if start + len(samples) <= 0:
            continue
        buffer[max(start, 0) : start + len(samples)] = samples[max(-start, 0) :]
    buffer /= 32768.0
    return buffer


def has_audio_stream(path: str | Path) -> bool:
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return False


def audio_samples(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode audio, returning an empty array for a file that cannot be opened.

    A file that opens but fails to decode raises, as that is worth reporting.
    """
    import av

    try:
        container = av.open(str(path))
    except Exception:
        return np.zeros(0, dtype=np.float32)
    with container:
        return _decode_audio(container, sample_rate)
