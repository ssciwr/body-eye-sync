"""Audio processing tab: run the speech pipeline over one input's audio.

One input is shown at a time, chosen from every input that carries audio -- the
separately recorded audio inputs and the video inputs' own tracks. The pipeline
editor beside the results edits the experiment's speech pipeline, which every
input shares, and its "Run" buttons run the steps in a background thread, with
the speech turns they find listed once each step finishes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pydantic import ValidationError

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import (
    DiarizationStep,
    SpeechPipeline,
    StepSpec,
    TranscriptionStep,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import SPEECH_STEPS, PipelineEditor
from body_eye_sync.gui.workers import DiarizationWorker, TranscriptionWorker
from body_eye_sync.media import has_audio_stream

_START, _END, _SPEAKER, _TEXT = range(4)
_COLUMNS = ["Start", "End", "Speaker", "Text"]


def _kind(data: Video | Audio) -> str:
    """What kind of input this is, as the chooser names it."""
    if isinstance(data, Audio):
        return "audio"
    return "glasses" if isinstance(data, GlassesVideo) else "fixed"


def _time_text(seconds: float) -> str:
    """A time on the recording's own clock, as ``m:ss.s``."""
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


@dataclass
class _StepRunner:
    """How to run one speech step: its worker and the plumbing it needs."""

    worker_cls: type
    # if the step is ready to run (e.g. transcription needs speech turns).
    ready: Callable[[Speech], bool]
    # clear any previous results for this step, given the step's arguments
    begin: Callable[[Speech, StepSpec], None]
    on_finished: Callable[[], None]


class AudioProcessingTab(BaseTab):
    """Show one input's speech results and run the speech pipeline over it."""

    title = "Audio processing"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)

        self._thread: threading.Thread | None = None
        self._worker: DiarizationWorker | TranscriptionWorker | None = None
        #: Remaining step types queued by "Run all"; consumed one at a time as
        #: each step finishes, so later steps see earlier steps' results.
        self._pending_steps: list[type] = []
        #: The inputs that carry audio, in the order the chooser lists them.
        self._audio_inputs: list[Video | Audio] = []
        #: Whether a file holds an audio stream, keyed by path: asking means
        #: opening the file, and the chooser asks again on every refresh.
        self._has_audio: dict[str, bool] = {}

        self.input_selector = QComboBox()
        self.input_selector.currentIndexChanged.connect(self._on_input_selected)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.turns_table = QTableWidget(0, len(_COLUMNS))
        self.turns_table.setHorizontalHeaderLabels(_COLUMNS)
        self.turns_table.verticalHeader().setVisible(False)
        self.turns_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.turns_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.turns_table.horizontalHeader()
        header.setSectionResizeMode(_TEXT, QHeaderView.ResizeMode.Stretch)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_run)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Recording:"))
        top_bar.addWidget(self.input_selector, stretch=1)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.summary_label, stretch=1)
        bottom_bar.addWidget(self.cancel_button)

        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addLayout(top_bar)
        results_layout.addWidget(self.turns_table, stretch=1)
        results_layout.addLayout(bottom_bar)
        results_side = QWidget()
        results_side.setLayout(results_layout)

        # The pipeline editor is the authority for the experiment's speech
        # pipeline. The splitter lets the user give it as much room as they want.
        self.pipeline_editor = PipelineEditor(SPEECH_STEPS)
        self.pipeline_editor.changed.connect(self._on_pipeline_edited)
        self.pipeline_editor.run_requested.connect(self._start_step)
        self.pipeline_editor.run_all_requested.connect(self._start_run_all)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(results_side)
        self.splitter.addWidget(self.pipeline_editor)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)

        # How to run each step: its worker, readiness check and bookkeeping.
        self._step_runners: dict[type, _StepRunner] = {
            DiarizationStep: _StepRunner(
                worker_cls=DiarizationWorker,
                ready=lambda speech: True,
                begin=lambda speech, step: speech.begin_diarization(
                    step.embeddings_per_speaker
                ),
                on_finished=self._on_diarization_finished,
            ),
            TranscriptionStep: _StepRunner(
                worker_cls=TranscriptionWorker,
                ready=lambda speech: speech.data is not None,
                begin=lambda speech, step: speech.begin_transcription(),
                on_finished=self._on_transcription_finished,
            ),
        }
        self.refresh()

    def refresh(self) -> None:
        """Re-list the inputs, keeping the shown one if it is still there."""
        if self._thread is not None:
            # A run drives the results it writes into; leave it be.
            return
        shown = self.input()
        self._audio_inputs = list(self._inputs().values())
        self.input_selector.blockSignals(True)
        self.input_selector.clear()
        for data in self._audio_inputs:
            self.input_selector.addItem(f"{data.id} ({_kind(data)})")
        index = next(
            (i for i, data in enumerate(self._audio_inputs) if data is shown),
            0 if self._audio_inputs else -1,
        )
        self.input_selector.setCurrentIndex(index)
        self.input_selector.blockSignals(False)
        self.input_selector.setEnabled(bool(self._audio_inputs))
        self._show_selected_input()

    def input(self) -> Video | Audio | None:
        """The input being shown, or ``None`` if the experiment has none."""
        index = self.input_selector.currentIndex()
        if 0 <= index < len(self._audio_inputs):
            return self._audio_inputs[index]
        return None

    def speech(self) -> Speech | None:
        """The shown input's speech results, whichever kind of input it is."""
        data = self.input()
        return None if data is None else data.speech

    def is_busy(self) -> bool:
        """Whether a speech step is currently running."""
        return self._thread is not None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _on_input_selected(self, _index: int) -> None:
        self._show_selected_input()

    def _show_selected_input(self) -> None:
        """List the chosen input's speech results and bind the editor to it."""
        self._refresh_results()
        self._bind_editor_to_pipeline()
        self._update_step_availability()

    def _audio_path(self) -> Path | None:
        """The file the speech steps would run over, if it carries any audio.

        A camera records audio alongside its video, so a video input's own file
        is what its speech results are computed from.
        """
        data = self.input()
        if data is None or data.path is None:
            return None
        path = data.path
        key = str(path)
        # Only a yes is remembered: a file that is not there yet, or is still
        # being copied in, is asked about again rather than written off for good.
        if not self._has_audio.get(key, False):
            self._has_audio[key] = has_audio_stream(path)
        return path if self._has_audio[key] else None

    def _refresh_results(self) -> None:
        """Fill the turns table and its summary from the shown input."""
        speech = self.speech()
        data = None if speech is None else speech.data
        self.turns_table.clearContents()
        self.turns_table.setRowCount(0 if data is None else len(data))
        if data is None:
            self.summary_label.setText(self._nothing_to_show())
            return
        # Turns no transcribed word landed in have no text, and turns have none
        # at all until transcription has run.
        texts = data["text"] if "text" in data.columns else None
        for row, turn in enumerate(data.itertuples(index=False)):
            text = "" if texts is None else texts.iloc[row]
            values = [
                _time_text(turn.start),
                _time_text(turn.end),
                str(int(turn.speaker)),
                "" if pd.isna(text) else str(text),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == _SPEAKER:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif column in (_START, _END):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.turns_table.setItem(row, column, item)
        self.summary_label.setText(self._summary(speech))

    def _nothing_to_show(self) -> str:
        """Why the table is empty: no input, no audio in it, or nothing run yet."""
        if self.input() is None:
            return "This experiment has no inputs."
        if self._audio_path() is None:
            return "This recording has no audio track."
        return "No speech results yet; run diarization."

    def _summary(self, speech: Speech) -> str:
        parts = [
            f"{len(speech.speakers)} speaker(s)",
            f"{len(speech.data)} speech turn(s)",
        ]
        if speech.words is not None:
            parts.append(f"{len(speech.words)} transcribed word(s)")
        return ", ".join(parts)

    def _pipeline(self) -> SpeechPipeline | None:
        """The experiment's speech pipeline, which every input shares."""
        return self.experiment.pipeline.speech

    def _bind_editor_to_pipeline(self) -> None:
        """Populate the pipeline editor from the experiment (or disable it)."""
        if self.input() is None:
            self.pipeline_editor.reset()
            self.pipeline_editor.setEnabled(False)
            return
        self.pipeline_editor.setEnabled(True)
        # An experiment with the speech pipeline switched off still gets an
        # editor, showing the defaults it would start from; it is only written
        # back into the experiment once something is actually edited.
        self.pipeline_editor.set_from(self._pipeline() or SpeechPipeline())

    def _on_pipeline_edited(self) -> None:
        """Adopt the editor's pipeline as the experiment's, when it is valid."""
        pipeline = self._pipeline()
        if pipeline is None:
            # Editing the speech settings turns the pipeline back on.
            pipeline = SpeechPipeline()
        try:
            self.pipeline_editor.apply_to(pipeline)
        except (ValidationError, ValueError):
            self.status_message.emit("Pipeline has invalid settings; not applied")
            return
        self.experiment.pipeline.speech = pipeline
        self.experiment_changed.emit()

    def _update_step_availability(self) -> None:
        """Enable each step's "Run" button (and "Run all") once its inputs are ready.

        Diarization needs a recording with audio in it; transcription is
        attributed to the speech turns diarization finds, so it waits for them.
        Whether a *run* is currently in progress is handled separately, by
        disabling the whole pipeline editor (see ``_set_running``).
        """
        speech = self.speech()
        has_audio = self._audio_path() is not None
        has_turns = has_audio and speech is not None and speech.data is not None
        self.pipeline_editor.set_run_enabled(DiarizationStep, has_audio)
        self.pipeline_editor.set_run_enabled(TranscriptionStep, has_turns)
        self.pipeline_editor.set_run_all_enabled(has_audio)

    def _step_config(self, step_type):
        """The editor's validated config for a step, or None (with an alert)."""
        try:
            return self.pipeline_editor.config_for(step_type)
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return None

    @Slot(object)
    def _start_step(self, step_type: type) -> None:
        """Run one speech step, using the editor's current arguments for it."""
        if self._thread is not None:
            return
        speech = self.speech()
        path = self._audio_path()
        runner = self._step_runners[step_type]
        if speech is None or path is None or not runner.ready(speech):
            self._pending_steps = []
            return
        step = self._step_config(step_type)
        if step is None:
            self._pending_steps = []
            return

        # Discards that step's previous results; keeps everything else (a
        # transcription pass keeps the speech turns it is attributed to).
        runner.begin(speech, step)
        self._begin_run()

        self._worker = runner.worker_cls(speech, path, step)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(runner.on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _start_run_all(self) -> None:
        """Run every enabled speech step in order, one after another."""
        if self._thread is not None:
            return
        try:
            steps = self.pipeline_editor.enabled_steps()
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self._pending_steps = [type(step) for step in steps]
        self._continue_run_all()

    def _continue_run_all(self) -> None:
        """Start the next step queued by "Run all", if any are left."""
        if self._pending_steps:
            self._start_step(self._pending_steps.pop(0))

    def _begin_run(self) -> None:
        """Shared start-up for the diarization and transcription runs."""
        self._set_running(True)
        # Weights are downloaded before any audio is processed, so show a busy
        # bar until the run reports how far along it is.
        self.progress_changed.emit(0, 0, "Downloading weights…")

    @Slot(float)
    def _on_progress(self, fraction: float) -> None:
        # Reporting a total turns the busy "downloading" bar into a determinate
        # one on its own.
        self.progress_changed.emit(
            round(100 * fraction), 100, f"{self._worker.operation_name}…"
        )

    def _cancel_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")

    @Slot()
    def _on_diarization_finished(self) -> None:
        speech = self.speech()
        self.status_message.emit(
            f"Diarization finished: {len(speech.speakers)} speakers over "
            f"{len(speech.data)} speech turns"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot()
    def _on_transcription_finished(self) -> None:
        speech = self.speech()
        n_words = 0 if speech.words is None else len(speech.words)
        self.status_message.emit(
            f"Transcription finished: {n_words} words over "
            f"{len(speech.data)} speech turns"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        # A failure stops a "Run all" chain rather than pressing on regardless.
        self._pending_steps = []
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(f"{self._worker.operation_name} failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        # Cancelling one step cancels the rest of a "Run all" chain too.
        self._pending_steps = []
        self.status_message.emit(f"{self._worker.operation_name} cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            # background thread has reported back; drop our references to it.
            self._thread = None
            self._worker = None
            # Re-read the experiment: refresh() bows out while a run is on, so
            # this is where any change made to the inputs meanwhile is picked up.
            self.refresh()
            # However it ended, the run changed the recording's results.
            self.experiment_changed.emit()
        self.input_selector.setEnabled(not running and bool(self._audio_inputs))
        self.pipeline_editor.setEnabled(not running and self.input() is not None)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        # The window locks the actions that would pull the experiment away.
        self.busy_changed.emit(running)
