"""Audio processing tab: transcribe the speech from an input's audio."""

from __future__ import annotations

import threading

from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
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
    SpeechPipeline,
    TranscriptionStep,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import AudioPlaybackWidget, SPEECH_STEPS, PipelineEditor
from body_eye_sync.gui.workers import TranscriptionWorker

_START, _END, _TEXT = range(3)
_COLUMNS = ["Start", "End", "Text"]


def _kind(data: Video | Audio) -> str:
    """What kind of input this is, as the chooser names it."""
    if isinstance(data, Audio):
        return "audio"
    return "glasses" if isinstance(data, GlassesVideo) else "fixed"


def _time_text(seconds: float) -> str:
    """A time on the recording's own clock, as ``m:ss.s``."""
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


class AudioProcessingTab(BaseTab):
    """Show one input's speech results and transcribe its audio."""

    title = "Audio processing"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)

        self._thread: threading.Thread | None = None
        self._worker: TranscriptionWorker | None = None
        self._pending_inputs: list[Video | Audio] = []
        self._recordings: list[Video | Audio] = []
        self._live_word_count = 0

        self.input_selector = QComboBox()
        self.input_selector.currentIndexChanged.connect(self._on_input_selected)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.audio_player = AudioPlaybackWidget()
        self.audio_player.position_changed.connect(self._highlight_transcript_at)
        self._highlighted_row = -1

        self.transcript_table = QTableWidget(0, len(_COLUMNS))
        self.transcript_table.setHorizontalHeaderLabels(_COLUMNS)
        self.transcript_table.verticalHeader().setVisible(False)
        self.transcript_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.transcript_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.transcript_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.transcript_table.cellDoubleClicked.connect(self._play_transcript_row)
        header = self.transcript_table.horizontalHeader()
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
        results_layout.addWidget(self.audio_player)
        results_layout.addWidget(self.transcript_table, stretch=1)
        results_layout.addLayout(bottom_bar)
        results_side = QWidget()
        results_side.setLayout(results_layout)

        self.pipeline_editor = PipelineEditor(SPEECH_STEPS)
        self.pipeline_editor.changed.connect(self._on_pipeline_edited)
        self.pipeline_editor.run_requested.connect(
            lambda _step_type: self._start_transcription()
        )
        self.pipeline_editor.run_all_requested.connect(self._start_run_all)
        self.transcription_checkbox = QCheckBox("Transcribe this experiment's speech")
        self.transcription_checkbox.toggled.connect(self._on_pipeline_toggled)
        self.pipeline_group = QGroupBox("Speech pipeline")
        pipeline_group_layout = QVBoxLayout(self.pipeline_group)
        pipeline_group_layout.addWidget(self.transcription_checkbox)
        pipeline_group_layout.addWidget(self.pipeline_editor)

        pipeline_layout = QVBoxLayout()
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.addWidget(self.pipeline_group)
        pipeline_layout.addStretch(1)
        pipeline_side = QWidget()
        pipeline_side.setLayout(pipeline_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(results_side)
        self.splitter.addWidget(pipeline_side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)

        self.refresh()

    def _inputs(self) -> dict[str, Video | Audio]:
        """The inputs that carry sound, the only ones with speech to transcribe."""
        return {
            input_id: data
            for input_id, data in super()._inputs().items()
            if data.has_audio_track()
        }

    def refresh(self) -> None:
        """Re-list the inputs, keeping the shown one if it is still there."""
        if self._thread is not None:
            return
        shown = self.input()
        self._recordings = list(self._inputs().values())
        self.input_selector.blockSignals(True)
        self.input_selector.clear()
        for data in self._recordings:
            self.input_selector.addItem(f"{data.id} ({_kind(data)})")
        index = next(
            (i for i, data in enumerate(self._recordings) if data is shown),
            0 if self._recordings else -1,
        )
        self.input_selector.setCurrentIndex(index)
        self.input_selector.blockSignals(False)
        self.input_selector.setEnabled(bool(self._recordings))
        self._show_selected_input()

    def input(self) -> Video | Audio | None:
        """The input being shown, or ``None`` if the experiment has none."""
        index = self.input_selector.currentIndex()
        if 0 <= index < len(self._recordings):
            return self._recordings[index]
        return None

    def speech(self) -> Speech | None:
        """The shown input's speech results, whichever kind of input it is."""
        data = self.input()
        return None if data is None else data.speech

    def is_busy(self) -> bool:
        """Whether transcription is currently running."""
        return self._thread is not None

    def _on_input_selected(self, _index: int) -> None:
        self._show_selected_input()

    def _show_selected_input(self) -> None:
        """List the chosen input's speech results and bind the editor to it."""
        data = self.input()
        if data is None:
            self.audio_player.clear()
        else:
            self.audio_player.load(data.path)
        self._refresh_results()
        self._bind_editor_to_pipeline()
        self._update_run_availability()

    def _refresh_results(self) -> None:
        """Fill the transcript table and its summary from the shown input."""
        speech = self.speech()
        data = None if speech is None else speech.data
        self.transcript_table.clearContents()
        self.transcript_table.setRowCount(0 if data is None else len(data))
        self._highlighted_row = -1
        if data is None:
            self.summary_label.setText(self._nothing_to_show())
            return
        for row, segment in enumerate(data.itertuples(index=False)):
            self._set_transcript_row(row, segment.start, segment.end, segment.text)
        self.summary_label.setText(self._summary(speech))

    def _set_transcript_row(
        self, row: int, start: float, end: float, text: str
    ) -> None:
        """Populate one row shared by loaded and live transcript segments."""
        alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        start_item = QTableWidgetItem(_time_text(start))
        start_item.setTextAlignment(alignment)
        # Playback highlighting needs the exact bounds, not the rounded text.
        start_item.setData(Qt.ItemDataRole.UserRole, (float(start), float(end)))
        end_item = QTableWidgetItem(_time_text(end))
        end_item.setTextAlignment(alignment)

        self.transcript_table.setItem(row, _START, start_item)
        self.transcript_table.setItem(row, _END, end_item)
        self.transcript_table.setItem(row, _TEXT, QTableWidgetItem(str(text)))

    @Slot(float)
    def _highlight_transcript_at(self, seconds: float) -> None:
        """Select the transcript segment containing the playback position."""
        row_at_position = -1
        for row in range(self.transcript_table.rowCount()):
            item = self.transcript_table.item(row, _START)
            bounds = None if item is None else item.data(Qt.ItemDataRole.UserRole)
            if bounds is not None and bounds[0] <= seconds <= bounds[1]:
                row_at_position = row
                break
        if row_at_position == self._highlighted_row:
            return
        self._highlighted_row = row_at_position
        self.transcript_table.clearSelection()
        if row_at_position >= 0:
            self.transcript_table.selectRow(row_at_position)
            self.transcript_table.scrollToItem(
                self.transcript_table.item(row_at_position, _TEXT),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    @Slot(int, int)
    def _play_transcript_row(self, row: int, _column: int) -> None:
        """Seek to a double-clicked segment and start or continue playback."""
        if self._thread is not None:
            return
        item = self.transcript_table.item(row, _START)
        bounds = None if item is None else item.data(Qt.ItemDataRole.UserRole)
        if bounds is None:
            return
        self.audio_player.seek(bounds[0])
        self.audio_player.play()

    def _nothing_to_show(self) -> str:
        """Why the transcript table is empty."""
        if self.input() is not None:
            return "No transcript yet; run transcription."
        if any(data.path is not None for data in self.experiment.inputs):
            return "None of this experiment's recordings carry audio."
        return "This experiment has no inputs."

    def _summary(self, speech: Speech) -> str:
        words = 0 if speech.words is None else len(speech.words)
        return f"{len(speech.data)} segment(s), {words} word(s)"

    def _bind_editor_to_pipeline(self) -> None:
        """Bind the checkbox and editor to the experiment's speech pipeline."""
        pipeline = self.experiment.pipeline.speech
        self.transcription_checkbox.blockSignals(True)
        self.transcription_checkbox.setChecked(pipeline is not None)
        self.transcription_checkbox.blockSignals(False)
        if pipeline is not None:
            self.pipeline_editor.set_from(pipeline)
        self.pipeline_editor.setEnabled(
            pipeline is not None and self.input() is not None
        )

    @Slot(bool)
    def _on_pipeline_toggled(self, enabled: bool) -> None:
        """Enable or disable transcription for the experiment."""
        if enabled:
            pipeline = SpeechPipeline()
            self.experiment.pipeline.speech = pipeline
            self.pipeline_editor.set_from(pipeline)
        else:
            self.experiment.pipeline.speech = None
            self._pending_inputs = []
        self.pipeline_editor.setEnabled(enabled and self.input() is not None)
        self._update_run_availability()
        self.experiment_changed.emit()

    def _on_pipeline_edited(self) -> None:
        """Adopt the editor's pipeline as the experiment's, when it is valid."""
        pipeline = self.experiment.pipeline.speech
        if pipeline is None:
            return
        try:
            self.pipeline_editor.apply_to(pipeline)
        except (ValidationError, ValueError):
            self.status_message.emit("Pipeline has invalid settings; not applied")
            return
        self.experiment_changed.emit()

    def _update_run_availability(self) -> None:
        """Enable the "Run" buttons when there are recordings to transcribe."""
        enabled = self.experiment.pipeline.speech is not None
        self.pipeline_editor.set_run_enabled(
            TranscriptionStep, enabled and self.input() is not None
        )
        self.pipeline_editor.set_run_all_enabled(enabled and bool(self._recordings))

    def _transcription_config(self) -> TranscriptionStep | None:
        """The editor's validated transcription settings, or ``None``."""
        if self.experiment.pipeline.speech is None:
            return None
        try:
            return self.pipeline_editor.config_for(TranscriptionStep)
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return None

    @Slot()
    def _start_transcription(self) -> None:
        """Transcribe the selected recording with the editor's settings."""
        if self._thread is not None:
            return
        data = self.input()
        if data is None:
            self._pending_inputs = []
            return
        settings = self._transcription_config()
        if settings is None:
            self._pending_inputs = []
            return

        speech = data.speech
        speech.begin_transcription()
        self._begin_run()

        self._worker = TranscriptionWorker(speech, data.path, settings)
        self._worker.progress.connect(self._on_progress)
        self._worker.new_frame.connect(self._on_new_segment)
        self._worker.finished.connect(self._on_transcription_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _start_run_all(self) -> None:
        """Transcribe every recording in chooser order."""
        if self._thread is not None:
            return
        if self._transcription_config() is None:
            self._pending_inputs = []
            return
        self._pending_inputs = list(self._recordings)
        self._continue_run_all()

    def _continue_run_all(self) -> None:
        """Select and transcribe the next recording queued by "Run all"."""
        while self._pending_inputs and self._thread is None:
            data = self._pending_inputs.pop(0)
            index = self._recordings.index(data)
            self.input_selector.setCurrentIndex(index)
            self._start_transcription()

    def _begin_run(self) -> None:
        """Prepare the tab for a transcription run."""
        self.audio_player.pause()
        self._set_running(True)
        self.transcript_table.clearContents()
        self.transcript_table.setRowCount(0)
        self._live_word_count = 0
        self.summary_label.setText("Waiting for transcript segments…")
        self.progress_changed.emit(0, 0, "Downloading weights…")

    @Slot(object)
    def _on_new_segment(self, segment) -> None:
        """Append one provisional Whisper segment while transcription runs."""
        scrollbar = self.transcript_table.verticalScrollBar()
        following = scrollbar.value() == scrollbar.maximum()
        row = self.transcript_table.rowCount()
        self.transcript_table.insertRow(row)
        self._set_transcript_row(row, segment.start, segment.end, segment.text)
        self._live_word_count += len(segment.words)
        self.summary_label.setText(
            f"{row + 1} segment(s), {self._live_word_count} word(s) — transcribing…"
        )
        if following:
            self.transcript_table.scrollToBottom()

    @Slot(float)
    def _on_progress(self, fraction: float) -> None:
        self.progress_changed.emit(round(100 * fraction), 100, "Transcription…")

    def _cancel_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")

    @Slot()
    def _on_transcription_finished(self) -> None:
        speech = self.speech()
        n_words = 0 if speech.words is None else len(speech.words)
        self.status_message.emit(
            f"Transcription finished: {n_words} words over {len(speech.data)} segments"
        )
        self._set_running(False)
        self._continue_run_all()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        self._pending_inputs = []
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Transcription failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self._pending_inputs = []
        self.status_message.emit("Transcription cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self._thread = None
            self._worker = None
            self.refresh()
            self.experiment_changed.emit()
        self.input_selector.setEnabled(not running and bool(self._recordings))
        self.audio_player.setEnabled(not running)
        self.transcription_checkbox.setEnabled(not running)
        self.pipeline_editor.setEnabled(
            not running
            and self.input() is not None
            and self.experiment.pipeline.speech is not None
        )
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        self.busy_changed.emit(running)
