import threading
import time

import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    Pipeline,
    SpeechPipeline,
    TranscriptionStep,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.audio_processing import AudioProcessingTab
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word

TRANSCRIPT = [
    TranscriptSegment(
        0.0,
        1.0,
        "hello there",
        [Word(0.1, 0.4, "hello", 0.9), Word(0.5, 0.9, "there", 0.9)],
    ),
    TranscriptSegment(1.5, 2.5, "hi", [Word(1.6, 2.4, "hi", 0.8)]),
    TranscriptSegment(3.0, 4.0, "good to see you", [Word(3.1, 3.9, "good", 0.7)]),
]


@pytest.fixture(autouse=True)
def fast_speech(monkeypatch):
    """Stand in for Whisper, which is far too slow for a test."""
    calls = {}

    def transcribe(audio_path, **kwargs):
        calls["transcribe"] = kwargs
        for segment in TRANSCRIPT:
            time.sleep(0.02)
            yield segment

    monkeypatch.setattr("body_eye_sync.pipeline.transcription.transcribe", transcribe)
    return calls


@pytest.fixture
def experiment(data_dir):
    """An experiment with an audio input and a video that carries audio."""
    return Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people-talking.mp4")
            ],
            audio=[
                AudioInput(id="mic1", path=data_dir / "three-people-conversation.opus")
            ],
        )
    )


@pytest.fixture
def tab(qtbot, experiment):
    tab = AudioProcessingTab(experiment)
    qtbot.addWidget(tab)
    # The audio input, rather than the video that happens to be listed first.
    tab.input_selector.setCurrentIndex(1)
    return tab


@pytest.fixture
def empty_tab(qtbot):
    """The tab as it is for an experiment with no inputs at all."""
    tab = AudioProcessingTab(Experiment(ExperimentConfig()))
    qtbot.addWidget(tab)
    return tab


def _run_button(tab, step_type):
    """The pipeline editor section's "Run" button for ``step_type``."""
    return next(
        s.run_button for s in tab.pipeline_editor._sections if s.step_type is step_type
    )


def _section(tab, step_type):
    return next(s for s in tab.pipeline_editor._sections if s.step_type is step_type)


def _transcribe(qtbot, tab):
    tab._start_step(TranscriptionStep)
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1],
            "start": [0.0, 2.5],
            "end": [1.75, 4.0],
            "text": ["hallo", "welt"],
        }
    )


def test_without_inputs_nothing_can_be_run(empty_tab):
    assert empty_tab.input() is None
    assert empty_tab.input_selector.count() == 0
    assert not empty_tab.input_selector.isEnabled()
    assert not empty_tab.pipeline_editor.isEnabled()
    assert not _run_button(empty_tab, TranscriptionStep).isEnabled()
    assert empty_tab.summary_label.text() == "This experiment has no inputs."


def test_every_input_that_carries_audio_is_listed(tab):
    assert [
        tab.input_selector.itemText(i) for i in range(tab.input_selector.count())
    ] == ["room (fixed)", "mic1 (audio)"]
    assert tab.input() is tab.experiment.audio[0]
    assert tab.pipeline_editor.isEnabled()
    assert _run_button(tab, TranscriptionStep).isEnabled()
    assert tab.summary_label.text() == "No transcript yet; run transcription."


def test_a_video_input_runs_over_its_own_audio_track(tab, data_dir):
    tab.input_selector.setCurrentIndex(0)

    assert tab.input() is tab.experiment.fixed_videos[0]
    assert tab.speech() is tab.experiment.fixed_videos[0].speech
    assert tab._audio_path() == data_dir / "three-people-talking.mp4"
    assert _run_button(tab, TranscriptionStep).isEnabled()


def test_an_input_without_an_audio_track_cannot_be_run(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
            ]
        )
    )
    tab = AudioProcessingTab(experiment)
    qtbot.addWidget(tab)

    assert tab._audio_path() is None
    assert not _run_button(tab, TranscriptionStep).isEnabled()
    assert not tab.pipeline_editor.run_all_button.isEnabled()
    assert tab.summary_label.text() == "This recording has no audio track."


def test_run_transcription_populates_the_results_and_the_table(qtbot, tab):
    assert tab.transcript_table.rowCount() == 0

    _transcribe(qtbot, tab)

    speech = tab.speech()
    assert speech.data["text"].tolist() == ["hello there", "hi", "good to see you"]
    assert len(speech.words) == 4
    assert tab.transcript_table.rowCount() == 3
    assert tab.transcript_table.item(1, 0).text() == "0:01.5"
    assert tab.transcript_table.item(1, 2).text() == "hi"
    assert tab.summary_label.text() == "3 segment(s), 4 word(s)"
    assert not tab.cancel_button.isVisibleTo(tab)


def test_transcript_segments_are_shown_as_they_arrive(qtbot, tab, monkeypatch):
    continue_run = threading.Event()

    def streaming_transcribe(audio_path, **kwargs):
        yield TRANSCRIPT[0]
        continue_run.wait(timeout=5)
        yield from TRANSCRIPT[1:]

    monkeypatch.setattr(
        "body_eye_sync.pipeline.transcription.transcribe", streaming_transcribe
    )

    tab._start_step(TranscriptionStep)
    try:
        qtbot.waitUntil(lambda: tab.transcript_table.rowCount() == 1, timeout=5000)

        assert tab.is_busy()
        assert tab.speech().data is None
        assert tab.transcript_table.item(0, 2).text() == "hello there"
        assert tab.summary_label.text() == "1 segment(s), 2 word(s) — transcribing…"
    finally:
        continue_run.set()

    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)
    assert tab.transcript_table.rowCount() == 3
    assert tab.summary_label.text() == "3 segment(s), 4 word(s)"


def test_run_all_runs_the_only_speech_step(qtbot, tab):
    tab.pipeline_editor.run_all_button.click()
    qtbot.waitUntil(
        lambda: not tab.is_busy() and tab.speech().words is not None, timeout=20000
    )

    assert tab.speech().data["text"].tolist() == [
        "hello there",
        "hi",
        "good to see you",
    ]
    assert tab._pending_steps == []


def test_editor_args_reach_the_worker(qtbot, tab, fast_speech):
    form = _section(tab, TranscriptionStep).form
    form._widgets["model_name"].setCurrentText("tiny")
    form._widgets["language"].clear()

    _transcribe(qtbot, tab)

    assert fast_speech["transcribe"]["model_name"] == "tiny"
    # An unset optional setting reaches the pipeline as None, not as "None".
    assert fast_speech["transcribe"]["language"] is None


def test_a_language_typed_into_the_form_is_used(qtbot, tab, fast_speech):
    _section(tab, TranscriptionStep).form._widgets["language"].setText("de")

    _transcribe(qtbot, tab)

    assert fast_speech["transcribe"]["language"] == "de"


def test_cached_results_are_shown_and_unlock_transcription(qtbot, experiment):
    experiment.audio[0].speech.set_data(_segments())

    tab = AudioProcessingTab(experiment)
    qtbot.addWidget(tab)
    tab.input_selector.setCurrentIndex(1)

    assert tab.transcript_table.rowCount() == 2
    assert _run_button(tab, TranscriptionStep).isEnabled()


def test_each_input_keeps_its_own_results(qtbot, tab):
    _transcribe(qtbot, tab)

    tab.input_selector.setCurrentIndex(0)

    assert tab.speech().data is None
    assert tab.transcript_table.rowCount() == 0
    # It has audio of its own, so it is ready to be transcribed in turn.
    assert _run_button(tab, TranscriptionStep).isEnabled()


def test_editing_the_pipeline_updates_the_experiment(tab):
    assert [type(s) for s in tab.experiment.pipeline.speech.steps] == [
        TranscriptionStep
    ]

    _section(tab, TranscriptionStep).form._widgets["beam_size"].setValue(2)

    assert tab.experiment.pipeline.speech.transcription.beam_size == 2


def test_editing_the_pipeline_switches_speech_back_on(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            audio=[
                AudioInput(id="mic1", path=data_dir / "three-people-conversation.opus")
            ],
            pipeline=Pipeline(speech=None),
        )
    )
    tab = AudioProcessingTab(experiment)
    qtbot.addWidget(tab)
    # Nothing is written into the experiment just by showing the defaults.
    assert experiment.pipeline.speech is None

    _section(tab, TranscriptionStep).form._widgets["beam_size"].setValue(3)

    assert experiment.pipeline.speech.transcription.beam_size == 3


def test_a_loaded_pipeline_populates_the_editor(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            audio=[
                AudioInput(id="mic1", path=data_dir / "three-people-conversation.opus")
            ],
            pipeline=Pipeline(
                speech=SpeechPipeline(
                    transcription=TranscriptionStep(beam_size=2),
                )
            ),
        )
    )
    tab = AudioProcessingTab(experiment)
    qtbot.addWidget(tab)

    steps = {type(s): s for s in tab.pipeline_editor.enabled_steps()}
    assert set(steps) == {TranscriptionStep}
    assert steps[TranscriptionStep].beam_size == 2


def test_invalid_step_settings_abort_the_run(tab, monkeypatch, fast_speech):
    shown = {}
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.audio_processing.QMessageBox.critical",
        lambda *a, **k: shown.setdefault("called", True),
    )

    # Every speech setting is a bounded widget, so settings the models would
    # reject have to be arranged rather than typed in.
    def invalid(self, step_type):
        raise ValueError("bad settings")

    monkeypatch.setattr(
        "body_eye_sync.gui.widgets.pipeline_editor.PipelineEditor.config_for", invalid
    )

    tab._start_step(TranscriptionStep)

    assert shown.get("called")
    assert not tab.is_busy()
    assert "transcribe" not in fast_speech
    assert tab.speech().data is None


def test_the_tab_is_locked_and_reports_progress_while_a_step_runs(qtbot, tab):
    busy = []
    progress = []
    tab.busy_changed.connect(busy.append)
    tab.progress_changed.connect(
        lambda value, maximum, label: progress.append((value, maximum, label))
    )

    tab._start_step(TranscriptionStep)

    assert not tab.input_selector.isEnabled()
    assert not tab.pipeline_editor.isEnabled()
    assert tab.cancel_button.isVisibleTo(tab)
    assert busy == [True]
    assert progress[0] == (0, 0, "Downloading weights…")

    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    assert tab.input_selector.isEnabled()
    assert tab.pipeline_editor.isEnabled()
    assert busy == [True, False]
    # Transcription reported how far into the recording it had reached.
    assert any(label == "Transcription…" for _, _, label in progress)


def test_finishing_a_step_reports_it(qtbot, tab):
    messages = []
    tab.status_message.connect(messages.append)

    _run_button(tab, TranscriptionStep).click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    assert messages == ["Transcription finished: 4 words over 3 segments"]


def test_a_finished_run_marks_the_experiment_as_changed(qtbot, tab):
    changed = []
    tab.experiment_changed.connect(lambda: changed.append(True))

    _transcribe(qtbot, tab)

    # The window needs to know there are results worth saving.
    assert changed


def test_cancelling_a_run_leaves_no_results(qtbot, tab):
    messages = []
    tab.status_message.connect(messages.append)

    tab._start_step(TranscriptionStep)
    tab.cancel_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    assert tab.speech().data is None
    assert tab.transcript_table.rowCount() == 0
    assert messages == ["Transcription cancelled"]


def test_a_partial_transcript_is_discarded_on_cancel(qtbot, tab):
    _transcribe(qtbot, tab)
    assert tab.speech().data is not None

    # A second pass that is cancelled partway leaves no half-transcript behind.
    tab._start_step(TranscriptionStep)
    tab.cancel_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    assert tab.speech().data is None
    assert tab.transcript_table.rowCount() == 0


def test_live_transcript_is_removed_when_cancelled(qtbot, tab, monkeypatch):
    continue_run = threading.Event()

    def streaming_transcribe(audio_path, **kwargs):
        yield TRANSCRIPT[0]
        continue_run.wait(timeout=5)
        yield TRANSCRIPT[1]

    monkeypatch.setattr(
        "body_eye_sync.pipeline.transcription.transcribe", streaming_transcribe
    )

    tab._start_step(TranscriptionStep)
    try:
        qtbot.waitUntil(lambda: tab.transcript_table.rowCount() == 1, timeout=5000)
        tab.cancel_button.click()
    finally:
        continue_run.set()

    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)
    assert tab.speech().data is None
    assert tab.transcript_table.rowCount() == 0
    assert tab.summary_label.text() == "No transcript yet; run transcription."


def test_a_failed_step_is_reported(qtbot, tab, monkeypatch):
    def failing_transcribe(audio_path, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - never reached, makes this a generator

    monkeypatch.setattr(
        "body_eye_sync.pipeline.transcription.transcribe", failing_transcribe
    )
    # Avoid blocking on the modal failure dialog ``_on_failed`` shows.
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.audio_processing.QMessageBox.exec", lambda self: None
    )

    tab.pipeline_editor.run_all_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    assert tab._pending_steps == []
    assert tab.speech().data is None


def test_a_run_in_progress_ignores_a_refresh(qtbot, tab, data_dir):
    tab._start_step(TranscriptionStep)
    tab.experiment.add_audio(
        AudioInput(id="mic2", path=data_dir / "three-people-glasses-1.opus")
    )

    tab.refresh()

    # Re-listing the inputs mid-run would pull the results being written to out
    # from under the worker.
    assert tab.input_selector.count() == 2
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    # Once it is over the tab re-reads the experiment, so the change is not lost.
    assert tab.input_selector.count() == 3
    assert tab.input() is tab.experiment.audio[0]
    assert tab.speech().data is not None


def test_set_experiment_switches_to_the_new_ones_inputs(tab, data_dir):
    other = Experiment(
        ExperimentConfig(
            audio=[
                AudioInput(id="mic9", path=data_dir / "three-people-glasses-1.opus")
            ],
        )
    )

    tab.set_experiment(other)

    assert tab.input() is other.audio[0]
    assert tab.input_selector.currentText() == "mic9 (audio)"


def test_shutdown_cancels_a_running_step(tab):
    tab._start_step(TranscriptionStep)

    tab.shutdown()

    assert tab._thread is None or not tab._thread.is_alive()
