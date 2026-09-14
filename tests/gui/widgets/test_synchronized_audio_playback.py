import pandas as pd
import pytest
from types import SimpleNamespace

from body_eye_sync.gui.widgets.synchronized_audio_playback import (
    SynchronizedAudioPlaybackWidget,
    active_speakers,
)


def _recording(name, rate=1.0):
    return SimpleNamespace(
        id=name,
        path=f"{name}.wav",
        timeline=SimpleNamespace(
            offset=0.0,
            rate=rate,
            to_experiment_time=lambda local: local * rate,
            to_local_time=lambda experiment: experiment / rate,
        ),
    )


def test_active_speakers_include_overlapping_accepted_turns():
    turns = pd.DataFrame(
        {
            "start": [1.0, 2.0, 5.0],
            "end": [3.0, 4.0, 6.0],
            "speaker": ["p1", "p2", "p1"],
        }
    )

    assert active_speakers(turns, 0.5) == set()
    assert active_speakers(turns, 1.5) == {"p1"}
    assert active_speakers(turns, 2.5) == {"p1", "p2"}
    assert active_speakers(turns, 3.0) == {"p2"}


def test_no_turns_means_no_recording_is_audible():
    assert active_speakers(None, 1.0) == set()
    assert active_speakers(pd.DataFrame(), 1.0) == set()


def test_aligned_rows_outline_only_current_speakers(qtbot, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.gui.widgets.synchronized_audio_playback.media_duration",
        lambda _path: 10.0,
    )
    widget = SynchronizedAudioPlaybackWidget()
    qtbot.addWidget(widget)
    widget.load([_recording("p1"), _recording("p2"), _recording("room")])
    widget.set_turns(
        pd.DataFrame(
            {
                "start": [1.0, 2.0],
                "end": [3.0, 4.0],
                "speaker": ["p1", "p2"],
            }
        )
    )

    # Media players remain lazy until the user actually presses Play.
    assert widget.recording_ids == ["p1", "p2", "room"]
    assert all(track.player is None for track in widget._tracks.values())
    assert widget.color_for("p1").name() == widget._tracks["p1"].row._color.name()
    assert (
        widget._tracks["p1"].row._graph._color.name()
        == widget._tracks["p1"].row._color.name()
    )
    assert widget.color_for("p1").alpha() == 255
    assert widget._tracks["p1"].row._graph._color.alpha() == 255
    assert widget.color_for("missing") is None

    widget.seek(1.5)
    assert widget._tracks["p1"].row._label.font().bold()
    assert not widget._tracks["p2"].row._label.font().bold()
    assert not widget._tracks["room"].row._label.font().bold()

    widget.seek(2.5)
    assert widget._tracks["p1"].row._label.font().bold()
    assert widget._tracks["p2"].row._label.font().bold()
    assert not widget._tracks["room"].row._label.font().bold()


def test_mute_background_controls_non_contributing_recordings(qtbot, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.gui.widgets.synchronized_audio_playback.media_duration",
        lambda _path: 10.0,
    )
    widget = SynchronizedAudioPlaybackWidget()
    qtbot.addWidget(widget)
    widget.load([_recording("p1"), _recording("p2"), _recording("room")])
    widget.set_turns(pd.DataFrame({"start": [1.0], "end": [3.0], "speaker": ["p1"]}))

    class Output:
        def __init__(self):
            self.muted = None

        def setMuted(self, muted):
            self.muted = muted

        def deleteLater(self):
            pass

    for track in widget._tracks.values():
        track.output = Output()

    assert widget.mute_background_checkbox.isChecked()
    widget.seek(1.5)
    assert not widget._tracks["p1"].output.muted
    assert widget._tracks["p2"].output.muted
    assert widget._tracks["room"].output.muted

    widget.mute_background_checkbox.setChecked(False)
    assert all(not track.output.muted for track in widget._tracks.values())

    widget.mute_background_checkbox.setChecked(True)
    assert not widget._tracks["p1"].output.muted
    assert widget._tracks["p2"].output.muted
    assert widget._tracks["room"].output.muted


def test_players_run_at_the_recordings_corrected_clock_rates(qtbot, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.gui.widgets.synchronized_audio_playback.media_duration",
        lambda _path: 10.0,
    )
    widget = SynchronizedAudioPlaybackWidget()
    qtbot.addWidget(widget)
    widget.load([_recording("slow-clock", rate=1.0001)])

    widget._ensure_players()

    player = widget._tracks["slow-clock"].player
    assert player is not None
    assert player.playbackRate() == pytest.approx(1.0 / 1.0001)
