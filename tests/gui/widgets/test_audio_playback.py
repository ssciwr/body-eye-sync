from pathlib import Path

import numpy as np

from body_eye_sync.gui.widgets.audio_playback import (
    AudioPlaybackWidget,
    _loudness_envelope,
)


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
