"""Measuring how loud a recording is, on synthetic audio of known loudness."""

import wave

import numpy as np
import pytest

from body_eye_sync.pipeline.loudness import (
    HOP_SECONDS,
    LOUDNESS_COLUMNS,
    measure_loudness,
)
from body_eye_sync.preprocessing.audio import SAMPLE_RATE

DURATION = 12.0


def _write(path, samples):
    """Write mono float samples to a 16-bit wav file."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def _speech(spans, seed=0):
    """Quiet noise throughout, with loud tone bursts over each ``(start, end)``."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.001, int(DURATION * SAMPLE_RATE))
    times = np.arange(samples.size) / SAMPLE_RATE
    for start, end in spans:
        span = (times >= start) & (times < end)
        samples[span] += 0.3 * np.sin(2 * np.pi * 220 * times[span])
    return samples


def test_loudness_follows_the_loudness_of_the_recording(tmp_path):
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)]))

    table = measure_loudness(tmp_path / "a.wav")

    assert list(table.columns) == LOUDNESS_COLUMNS
    assert len(table) == pytest.approx(DURATION / HOP_SECONDS, abs=1)
    levels = table["level_db"].to_numpy()
    assert levels[int(2.0 / HOP_SECONDS)] > levels[int(10.0 / HOP_SECONDS)] + 20


def test_every_row_is_timed_at_the_middle_of_the_chunk_it_covers(tmp_path):
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)]))

    times = measure_loudness(tmp_path / "a.wav")["time"].to_numpy()

    assert times[0] == pytest.approx(HOP_SECONDS / 2)
    assert np.diff(times) == pytest.approx(HOP_SECONDS)


def test_a_recording_too_short_to_measure_has_no_rows(tmp_path):
    _write(tmp_path / "blip.wav", np.zeros(10))

    table = measure_loudness(tmp_path / "blip.wav")

    assert list(table.columns) == LOUDNESS_COLUMNS
    assert table.empty
