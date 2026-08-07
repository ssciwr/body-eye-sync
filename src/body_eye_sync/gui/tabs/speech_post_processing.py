"""Speech post processing tab: who spoke when, across the whole experiment."""

from __future__ import annotations

import threading
import traceback

from qtpy.QtCore import QItemSelectionModel, QObject, Qt, Signal, Slot
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
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

from body_eye_sync.experiment.config import SpeechPostProcessingSettings
from body_eye_sync.experiment.postprocess import attribute_experiment_speech
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import PydanticForm, SynchronizedAudioPlaybackWidget
from body_eye_sync.postprocessing.attribution import AttributionCancelled
from body_eye_sync.preprocessing.audio import has_audio_stream

_START, _END, _SPEAKER, _TEXT = range(4)
_COLUMNS = ["Start", "End", "Speaker", "Text"]

_LABEL = "Attributing speech…"

_SPLITTING_FIELDS = (
    "split_gap_seconds",
    "split_on_sentence_end",
    "split_on_comma",
    "minimum_clause_words",
    "minimum_clause_seconds",
)
_ATTRIBUTION_FIELDS = (
    "floor_percentile",
    "live_above_floor_db",
    "ownership_share",
    "fuzzy_agreement",
)


def _time_text(seconds: float) -> str:
    """A moment of the experiment, as ``m:ss.s``."""
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


class _Worker(QObject):
    """Run the attribution pass without blocking Qt's event loop."""

    progress = Signal(int)
    finished = Signal()
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self._experiment = experiment
        self._cancel = threading.Event()
        self._reported = -1

    def cancel(self) -> None:
        self._cancel.set()

    def _progress(self, value: float) -> bool:
        percent = round(100 * value)
        if percent != self._reported:
            self._reported = percent
            self.progress.emit(percent)
        return not self._cancel.is_set()

    @Slot()
    def run(self) -> None:
        try:
            attribute_experiment_speech(self._experiment, progress=self._progress)
        except AttributionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit()


class SpeechPostProcessingTab(BaseTab):
    """Work out the experiment's speech turns from its glasses recordings."""

    title = "Speech post processing"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self._thread: threading.Thread | None = None
        self._worker: _Worker | None = None
        self._has_audio: dict[str, bool] = {}

        self.attribute_button = QPushButton("Attribute speech to speakers")
        self.attribute_button.clicked.connect(self._start)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.audio_player = SynchronizedAudioPlaybackWidget()
        self.audio_player.position_changed.connect(self._highlight_turns_at)
        self._highlighted_rows: set[int] = set()

        self.turns_table = QTableWidget(0, len(_COLUMNS))
        self.turns_table.setHorizontalHeaderLabels(_COLUMNS)
        self.turns_table.verticalHeader().setVisible(False)
        self.turns_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.turns_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.turns_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.turns_table.cellDoubleClicked.connect(self._play_turn)
        self.turns_table.horizontalHeader().setSectionResizeMode(
            _TEXT, QHeaderView.ResizeMode.Stretch
        )

        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addWidget(self.audio_player)
        results_layout.addWidget(self.turns_table, stretch=1)
        results_layout.addWidget(self.summary_label)
        results_side = QWidget()
        results_side.setLayout(results_layout)

        settings = self.experiment.pipeline.speech_post_processing
        self.splitting_form = PydanticForm(settings, fields=_SPLITTING_FIELDS)
        self.splitting_form.changed.connect(self._on_settings_changed)
        self.splitting_group = QGroupBox("Splitting")
        splitting_group_layout = QVBoxLayout(self.splitting_group)
        splitting_group_layout.addWidget(self.splitting_form)

        self.attribution_form = PydanticForm(settings, fields=_ATTRIBUTION_FIELDS)
        self.attribution_form.changed.connect(self._on_settings_changed)
        self.attribution_group = QGroupBox("Attribution")
        attribution_group_layout = QVBoxLayout(self.attribution_group)
        attribution_group_layout.addWidget(self.attribution_form)

        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(self.splitting_group)
        settings_layout.addWidget(self.attribution_group)
        settings_layout.addWidget(self.attribute_button)
        settings_layout.addWidget(self.cancel_button)
        settings_layout.addStretch(1)
        settings_side = QWidget()
        settings_side.setLayout(settings_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(results_side)
        self.splitter.addWidget(settings_side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        self.refresh()

    def refresh(self) -> None:
        if self._thread is not None:
            return
        for form in (self.splitting_form, self.attribution_form):
            form.blockSignals(True)
            form.from_model(self.experiment.pipeline.speech_post_processing)
            form.blockSignals(False)
        self._refresh_audio()
        self._refresh_table()
        blocked = self.blocked_reason()
        self.attribute_button.setEnabled(blocked is None)
        lines = [line for line in (self._summary(), blocked) if line]
        self.summary_label.setText("\n".join(lines))

    def blocked_reason(self) -> str | None:
        """Why attribution cannot run yet, or ``None`` when it can."""
        glasses = [v for v in self.experiment.glasses_videos if v.path is not None]
        if len(glasses) < 2:
            return (
                "Speaker attribution compares the glasses recordings against each "
                "other, so it needs at least two of them."
            )
        missing = sorted(v.id for v in glasses if v.speech.data is None)
        if missing:
            return (
                "Transcribe these recordings first, in the Audio processing tab: "
                + ", ".join(missing)
            )
        if not any(v.timeline.offset for v in glasses):
            return (
                "Align the recordings first, in the Alignment tab: attribution "
                "compares them moment by moment, so it needs them on one clock."
            )
        return None

    def is_busy(self) -> bool:
        return self._thread is not None

    @Slot()
    def _on_settings_changed(self) -> None:
        """Persist the visible settings as the experiment's post-processing config."""
        settings = self.experiment.pipeline.speech_post_processing
        settings = self.splitting_form.to_model(settings)
        settings = self.attribution_form.to_model(settings)
        if not isinstance(settings, SpeechPostProcessingSettings):
            return
        self.experiment.pipeline.speech_post_processing = settings
        self.experiment_changed.emit()

    def _refresh_table(self) -> None:
        """Show the experiment's speech turns, however they got there."""
        turns = self.experiment.speech_turns
        data = turns.data
        self.turns_table.clearContents()
        self.turns_table.setRowCount(0 if data is None else len(data))
        self._highlighted_rows.clear()
        if data is None:
            return
        for row, turn in enumerate(data.itertuples(index=False)):
            color = self.audio_player.color_for(str(turn.speaker))
            background = None
            if color is not None:
                background = QColor(color)
                background.setAlpha(48)
            values = [
                _time_text(turn.start),
                _time_text(turn.end),
                str(turn.speaker),
                str(turn.text),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (_START, _END):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if background is not None:
                    item.setBackground(QBrush(background))
                self.turns_table.setItem(row, column, item)
            self.turns_table.item(row, _START).setData(
                Qt.ItemDataRole.UserRole, (float(turn.start), float(turn.end))
            )

    def _refresh_audio(self) -> None:
        """Show every synchronized input that carries an audio stream."""
        recordings = []
        for data in self.experiment.inputs:
            if data.path is None:
                continue
            key = str(data.path)
            if not self._has_audio.get(key, False):
                self._has_audio[key] = has_audio_stream(data.path)
            if self._has_audio[key]:
                recordings.append(data)
        self.audio_player.load(recordings)
        self.audio_player.set_turns(self.experiment.speech_turns.data)

    @Slot(int, int)
    def _play_turn(self, row: int, _column: int) -> None:
        """Seek to a double-clicked accepted turn and begin shared playback."""
        item = self.turns_table.item(row, _START)
        bounds = None if item is None else item.data(Qt.ItemDataRole.UserRole)
        if bounds is None:
            return
        self.audio_player.seek(bounds[0])
        self.audio_player.play()

    @Slot(float)
    def _highlight_turns_at(self, seconds: float) -> None:
        """Select every accepted turn containing the shared playback position."""
        rows = set()
        for row in range(self.turns_table.rowCount()):
            item = self.turns_table.item(row, _START)
            bounds = None if item is None else item.data(Qt.ItemDataRole.UserRole)
            if bounds is not None and bounds[0] <= seconds < bounds[1]:
                rows.add(row)
        if rows == self._highlighted_rows:
            return

        self._highlighted_rows = rows
        self.turns_table.clearSelection()
        selection = self.turns_table.selectionModel()
        flags = (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        )
        for row in sorted(rows):
            selection.select(self.turns_table.model().index(row, 0), flags)
        if rows:
            first_row = min(rows)
            self.turns_table.scrollToItem(
                self.turns_table.item(first_row, _TEXT),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    def _summary(self) -> str:
        """What the table holds, or nothing at all when it holds nothing."""
        turns = self.experiment.speech_turns
        data = turns.data
        if data is None:
            return ""
        if data.empty:
            return "No speech was attributed to anyone."
        words = int(data["text"].str.split().str.len().sum())
        per_speaker = ", ".join(
            f"{speaker} {len(turns.for_speaker(speaker))}" for speaker in turns.speakers
        )
        return (
            f"{len(data)} turn(s), {words} word(s) across "
            f"{len(turns.speakers)} speaker(s) — turns each: {per_speaker}"
        )

    def _start(self) -> None:
        if self._thread is not None or self.blocked_reason() is not None:
            return
        self.summary_label.setText("Measuring how loud each recording is…")
        self._set_running(True)
        self.progress_changed.emit(0, 100, _LABEL)
        self._worker = _Worker(self.experiment)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")

    @Slot(int)
    def _on_progress(self, percent: int) -> None:
        self.progress_changed.emit(percent, 100, _LABEL)

    @Slot()
    def _on_finished(self) -> None:
        turns = self.experiment.speech_turns
        count = 0 if turns.data is None else len(turns.data)
        self.status_message.emit(
            f"Attributed {count} speech turns across {len(turns.speakers)} speakers"
        )
        self._set_running(False)
        self.experiment_changed.emit()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Speaker attribution failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self._set_running(False)
        self.summary_label.setText("Could not attribute the speech.")

    @Slot()
    def _on_cancelled(self) -> None:
        self.status_message.emit("Speaker attribution cancelled")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self._thread = None
            self._worker = None
            self.refresh()
        self.attribute_button.setEnabled(not running and self.blocked_reason() is None)
        self.splitting_group.setEnabled(not running)
        self.attribution_group.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        self.busy_changed.emit(running)
