from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_audio(audio_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio_path), sampling_rate=sample_rate)


def has_audio_stream(path: str | Path) -> bool:
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return False


def audio_samples(path: str | Path, sample_rate: int) -> np.ndarray:
    if not has_audio_stream(path):
        return np.zeros(0, dtype=np.float32)
    return np.asarray(load_audio(path, sample_rate), dtype=np.float32)
