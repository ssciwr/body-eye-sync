import wave

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.speech_post_processing import SpeechPostProcessingTab
from body_eye_sync.media import SAMPLE_RATE

DURATION = 12.0


def _write(path, spans, gain=1.0, seed=0):
    """A recording that is quiet throughout except for loud bursts over ``spans``."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.001, int(DURATION * SAMPLE_RATE))
    times = np.arange(samples.size) / SAMPLE_RATE
    for start, end in spans:
        span = (times >= start) & (times < end)
        samples[span] += gain * 0.3 * np.sin(2 * np.pi * 220 * times[span])
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _transcript(*rows):
    return pd.DataFrame(
        [(index, *row) for index, row in enumerate(rows)],
        columns=["segment_id", "start", "end", "text"],
    )


@pytest.fixture
def experiment(tmp_path):
    """Two aligned, transcribed wearers who take turns speaking."""
    _write(tmp_path / "p1.wav", [(1.0, 4.0)], seed=1)
    _write(tmp_path / "p2.wav", [(6.0, 9.0)], gain=0.5, seed=2)
    gaze = tmp_path / "gaze.tsv"
    gaze.write_text("")
    exp = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(id="p1", path=tmp_path / "p1.wav", gaze_path=gaze),
                GlassesVideoInput(
                    id="p2",
                    path=tmp_path / "p2.wav",
                    gaze_path=gaze,
                    time_offset=0.001,  # aligned, even if barely
                ),
            ]
        ),
        tmp_path / "experiment",
    )
    exp.glasses_videos[0].speech.set_data(
        _transcript((1.0, 4.0, "p1 speaking"), (6.0, 9.0, "p2 as p1 heard them"))
    )
    exp.glasses_videos[1].speech.set_data(
        _transcript((1.0, 4.0, "p1 as p2 heard them"), (6.0, 9.0, "p2 speaking"))
    )
    return exp


@pytest.fixture
def tab(qtbot, experiment):
    tab = SpeechPostProcessingTab(experiment)
    qtbot.addWidget(tab)
    return tab


def _attribute(qtbot, tab):
    tab.attribute_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=30000)


def test_a_ready_experiment_can_be_attributed(tab):
    assert tab.blocked_reason() is None
    assert tab.attribute_button.isEnabled()


def test_attribution_fills_the_turns_table(qtbot, tab):
    assert tab.turns_table.rowCount() == 0

    _attribute(qtbot, tab)

    assert tab.turns_table.rowCount() == 2
    assert tab.turns_table.item(0, 2).text() == "p1"
    assert tab.turns_table.item(0, 3).text() == "p1 speaking"
    assert tab.turns_table.item(1, 2).text() == "p2"
    assert tab.turns_table.item(1, 3).text() == "p2 speaking"
    assert "2 turn(s)" in tab.summary_label.text()
    assert "2 speaker(s)" in tab.summary_label.text()


def test_the_turns_are_stored_on_the_experiment(qtbot, tab):
    changed = []
    tab.experiment_changed.connect(lambda: changed.append(True))

    _attribute(qtbot, tab)

    assert tab.experiment.speech_turns.speakers == ["p1", "p2"]
    # The window needs to know there are results worth saving.
    assert changed


def test_stored_turns_are_shown_without_running_anything(qtbot, experiment):
    experiment.speech_turns.set_data(
        pd.DataFrame(
            {
                "turn_id": [0],
                "start": [1.0],
                "end": [4.0],
                "speaker": ["p1"],
                "source": ["p1"],
                "source_segment_id": [0],
                "text": ["from an earlier run"],
            }
        )
    )

    tab = SpeechPostProcessingTab(experiment)
    qtbot.addWidget(tab)

    assert tab.turns_table.rowCount() == 1
    assert tab.turns_table.item(0, 3).text() == "from an earlier run"


def test_one_glasses_recording_cannot_be_attributed(qtbot, tmp_path):
    _write(tmp_path / "p1.wav", [(1.0, 4.0)])
    gaze = tmp_path / "gaze.tsv"
    gaze.write_text("")
    exp = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(id="p1", path=tmp_path / "p1.wav", gaze_path=gaze)
            ]
        )
    )
    tab = SpeechPostProcessingTab(exp)
    qtbot.addWidget(tab)

    assert "at least two" in tab.blocked_reason()
    assert not tab.attribute_button.isEnabled()


def test_untranscribed_recordings_are_named_in_the_reason(tab):
    tab.experiment.glasses_videos[1].speech.clear()
    tab.refresh()

    reason = tab.blocked_reason()
    assert "Transcribe these recordings first" in reason
    assert "p2" in reason
    assert not tab.attribute_button.isEnabled()


def test_unaligned_recordings_are_refused(tab):
    # Every offset still at zero: the recordings have never been aligned, and
    # comparing them moment by moment would attribute speech to the wrong people.
    for video in tab.experiment.glasses_videos:
        video.time_offset = 0.0
    tab.refresh()

    assert "Align the recordings first" in tab.blocked_reason()
    assert not tab.attribute_button.isEnabled()


def test_a_fixed_camera_does_not_make_it_runnable(qtbot, tmp_path):
    _write(tmp_path / "p1.wav", [(1.0, 4.0)])
    _write(tmp_path / "room.wav", [(1.0, 9.0)])
    gaze = tmp_path / "gaze.tsv"
    gaze.write_text("")
    exp = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(id="p1", path=tmp_path / "p1.wav", gaze_path=gaze)
            ],
            fixed_videos=[FixedVideoInput(id="room", path=tmp_path / "room.wav")],
        )
    )
    tab = SpeechPostProcessingTab(exp)
    qtbot.addWidget(tab)

    # Nobody wears a room camera, so it cannot stand in for a second wearer.
    assert "at least two" in tab.blocked_reason()


def test_the_tab_reports_being_busy_and_shows_progress(qtbot, tab):
    busy = []
    progress = []
    tab.busy_changed.connect(busy.append)
    tab.progress_changed.connect(
        lambda value, maximum, label: progress.append((value, maximum, label))
    )

    tab.attribute_button.click()

    assert tab.is_busy()
    assert tab.cancel_button.isVisibleTo(tab)
    assert busy == [True]

    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=30000)

    assert busy == [True, False]
    assert progress[0] == (0, 100, "Attributing speech…")
    assert progress[-1] == (100, 100, "Attributing speech…")
    assert not tab.cancel_button.isVisibleTo(tab)


def test_finishing_reports_what_was_found(qtbot, tab):
    messages = []
    tab.status_message.connect(messages.append)

    _attribute(qtbot, tab)

    assert messages == ["Attributed 2 speech turns across 2 speakers"]


def test_a_failed_run_is_reported(qtbot, tab, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.speech_post_processing.attribute_experiment_speech",
        boom,
    )
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.speech_post_processing.QMessageBox.exec",
        lambda self: None,
    )

    tab.attribute_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=30000)

    assert tab.summary_label.text() == "Could not attribute the speech."
    assert tab.experiment.speech_turns.data is None


def test_set_experiment_shows_the_new_ones_turns(tab):
    tab.set_experiment(Experiment(ExperimentConfig()))

    assert tab.turns_table.rowCount() == 0
    assert not tab.attribute_button.isEnabled()


def test_shutdown_stops_a_running_pass(tab):
    tab.attribute_button.click()

    tab.shutdown()

    assert tab._thread is None or not tab._thread.is_alive()
