"""How loud a recording is over time, measured from its audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from body_eye_sync.preprocessing.audio import SAMPLE_RATE, load_audio

# length of the audio chunk each loudness value covers.
HOP_SECONDS = 0.05

LOUDNESS_COLUMNS = ["time", "level_db"]


def measure_loudness(path: str | Path) -> pd.DataFrame:
    """How loud a recording is over time, one row every :data:`HOP_SECONDS`.

    ``level_db`` is the chunk's RMS level in dB and ``time`` is its middle, on
    the recording's own clock. A recording too short to fill a single chunk
    measures as no rows at all.
    """
    samples = load_audio(path, SAMPLE_RATE)
    frame = int(HOP_SECONDS * SAMPLE_RATE)
    if samples.size < frame:
        level_db = np.empty(0)
    else:
        frames = samples[: samples.size // frame * frame].reshape(-1, frame)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        level_db = 20 * np.log10(np.maximum(rms, 1e-8))
    return pd.DataFrame(
        {
            "time": np.arange(level_db.size) * HOP_SECONDS + HOP_SECONDS / 2,
            "level_db": level_db,
        }
    )
