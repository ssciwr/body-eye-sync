"""Speech post processing tab: who spoke when, across the whole experiment.

The per-input passes transcribe each recording on its own, which says what was
said but never who said it. This stage relates the recordings to each other:
each glasses recording belongs to a known wearer, and whoever's microphone was
loudest is whoever was speaking. It therefore acts on the experiment rather than
on a selected input, and needs the recordings to have been aligned and
transcribed first.
"""

from __future__ import annotations

import threading
import traceback

from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from body_eye_sync.experiment.attribution import attribute_experiment_speech
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.preprocessing.attribution import AttributionCancelled

_START, _END, _SPEAKER, _TEXT = range(4)
_COLUMNS = ["Start", "End", "Speaker", "Text"]

_LABEL = "Attributing speech…"


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

        self.attribute_button = QPushButton("Attribute speech to speakers")
        self.attribute_button.clicked.connect(self._start)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.turns_table = QTableWidget(0, len(_COLUMNS))
        self.turns_table.setHorizontalHeaderLabels(_COLUMNS)
        self.turns_table.verticalHeader().setVisible(False)
        self.turns_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.turns_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.turns_table.horizontalHeader().setSectionResizeMode(
            _TEXT, QHeaderView.ResizeMode.Stretch
        )

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.attribute_button)
        top_bar.addWidget(self.cancel_button)
        top_bar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self.turns_table, stretch=1)
        layout.addWidget(self.summary_label)
        self.refresh()

    def refresh(self) -> None:
        if self._thread is not None:
            # A run owns the experiment's turns; leave them be until it ends.
            return
        self._refresh_table()
        blocked = self.blocked_reason()
        self.attribute_button.setEnabled(blocked is None)
        # Turns already worked out are still worth summarising, even when they
        # could not be worked out again; both are said when both apply.
        lines = [line for line in (self._summary(), blocked) if line]
        self.summary_label.setText("\n".join(lines))

    def blocked_reason(self) -> str | None:
        """Why attribution cannot run yet, or ``None`` when it can.

        Attribution compares the glasses recordings against each other, so it
        needs at least two of them, each transcribed, and all of them placed on
        the same clock. Running without those produces confident nonsense rather
        than an obvious failure, so it is refused instead.
        """
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
        if not any(v.time_offset for v in glasses):
            return (
                "Align the recordings first, in the Alignment tab: attribution "
                "compares them moment by moment, so it needs them on one clock."
            )
        return None

    def is_busy(self) -> bool:
        return self._thread is not None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _refresh_table(self) -> None:
        """Show the experiment's speech turns, however they got there."""
        turns = self.experiment.speech_turns
        data = turns.data
        self.turns_table.clearContents()
        self.turns_table.setRowCount(0 if data is None else len(data))
        if data is None:
            return
        for row, turn in enumerate(data.itertuples(index=False)):
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
                self.turns_table.setItem(row, column, item)

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
        self._worker.progress.connect(
            lambda value: self.progress_changed.emit(value, 100, _LABEL)
        )
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
        # After the refresh, which would otherwise overwrite what it says.
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
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        self.busy_changed.emit(running)
