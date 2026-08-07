"""Reading recordings: the bits every stage needs before it can do anything.

Decoding audio and asking a container what it holds is neither preprocessing nor
pipeline work, so it lives here rather than in either, and both may use it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The sample rate the audio models expect.
SAMPLE_RATE = 16000


def load_audio(audio_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a recording to mono ``float32`` samples at ``sample_rate``."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio_path), sampling_rate=sample_rate)


def has_audio_stream(path: str | Path) -> bool:
    """Whether a media file carries an audio track."""
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return False


def media_duration(path: str | Path) -> float | None:
    """How long a recording runs, read from its container without decoding it.

    ``None`` when the container does not say and no stream does either, which a
    caller has to allow for: some formats only reveal it by being read through.
    """
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
