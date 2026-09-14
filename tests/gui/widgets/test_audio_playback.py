from pathlib import Path
import wave

import numpy as np
import pytest

from body_eye_sync.gui.widgets.audio_playback import (
    AudioPlaybackWidget,
    _loudness_envelope,
    loudness_overview,
)
from body_eye_sync.pipeline.loudness import measure_loudness
from body_eye_sync.preprocessing.audio import SAMPLE_RATE


def test_loudness_envelope_keeps_quiet_and_loud_sections_visible(monkeypatch):
    samples = np.concatenate(
        [np.full(100, 0.01, dtype=np.float32), np.full(100, 0.5, dtype=np.float32)]
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.audio.load_audio", lambda _path: samples
    )

    envelope = _loudness_envelope(Path("recording.wav"), points=2)

    assert envelope.shape == (2,)
    assert 0 < envelope[0] < envelope[1]
    assert envelope[1] == 1


def test_loudness_envelope_draws_silence_at_zero(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.audio.load_audio",
        lambda _path: np.zeros(100, dtype=np.float32),
    )

    assert _loudness_envelope(Path("silence.wav"), points=10).tolist() == [0] * 10


def test_measured_levels_draw_the_overview_decoding_would_give(tmp_path):
    # Ten seconds of quiet with a loud burst, measured the way audio
    # processing measures every recording it transcribes.
    times = np.arange(10 * SAMPLE_RATE) / SAMPLE_RATE
    samples = np.full(times.size, 0.002)
    samples[(times >= 3.0) & (times < 6.0)] += (
        0.3 * np.sin(2 * np.pi * 220 * times)[(times >= 3.0) & (times < 6.0)]
    )
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

    overview = loudness_overview(measure_loudness(path)["level_db"], points=200)

    assert overview == pytest.approx(_loudness_envelope(path, points=200), abs=1e-6)


def test_a_measured_recording_is_drawn_without_decoding_it(qtbot, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.audio.load_audio",
        lambda *args, **kwargs: pytest.fail("decoded a recording already measured"),
    )
    widget = AudioPlaybackWidget()
    qtbot.addWidget(widget)

    widget.load(Path("recording.wav"), np.array([-60.0, -20.0, -30.0, -50.0]))
    widget.show()

    assert widget._graph._values.size == 4


def test_player_updates_seek_controls_and_emits_seconds(qtbot, tmp_path):
    widget = AudioPlaybackWidget()
    qtbot.addWidget(widget)
    positions = []
    widget.position_changed.connect(positions.append)

    widget.load(tmp_path / "recording.wav")
    widget._on_duration_changed(90_000)
    widget._on_position_changed(12_300)

    assert widget._controls.slider.maximum() == 90_000
    assert widget._controls.slider.value() == 12_300
    assert widget._controls.time_label.text() == "0:12.3 / 1:30.0"
    assert positions[-1] == 12.3

    widget._seek_fraction(0.5)
    assert widget._controls.slider.value() == 45_000
    assert positions[-1] == 45.0
