"""Data export tab: write a synchronized combined video."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.export.elan import export_elan
from body_eye_sync.export.video_grid import (
    VideoGridCancelled,
    VideoGridResult,
    construct_video_grid,
)
from body_eye_sync.gui.tabs.base import BaseTab

_INPUT_ID_ROLE = Qt.ItemDataRole.UserRole
_IS_VIDEO_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def _input_kind(data: Video | Audio) -> str:
    if isinstance(data, GlassesVideo):
        return "glasses video"
    if isinstance(data, Video):
        return "fixed video"
    return "audio"


class _VideoExportWorker(QObject):
    """Construct one video without blocking Qt's event loop."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(
        self,
        experiment: Experiment,
        output_path: Path,
        input_ids: list[str],
        include_merged_audio: bool,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._output_path = output_path
        self._input_ids = input_ids
        self._include_merged_audio = include_merged_audio
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
            result = construct_video_grid(
                self._experiment,
                self._output_path,
                input_ids=self._input_ids,
                include_merged_audio=self._include_merged_audio,
                overwrite=True,
                progress=self._progress,
            )
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            export_elan(
                self._experiment,
                result,
                input_ids=self._input_ids,
                overwrite=True,
            )
        except VideoGridCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit(result)


class DataExportTab(BaseTab):
    """Choose experiment inputs and export their synchronized video grid."""

    title = "Data export"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self._thread: threading.Thread | None = None
        self._worker: _VideoExportWorker | None = None

        description = QLabel(
            "Select the inputs to include in the synchronized 25 fps video. "
            "Video inputs become grid cells; audio-only inputs contribute audio "
            "tracks. A matching ELAN file contains available speech turns."
        )
        description.setWordWrap(True)

        self.input_list = QListWidget()
        self.input_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.input_list.setAlternatingRowColors(True)
        self.input_list.itemChanged.connect(self._update_availability)

        self.merged_audio_checkbox = QCheckBox("Include merged audio track")
        self.merged_audio_checkbox.setToolTip(
            "Append one default playback track mixing the synchronized audio from "
            "all selected inputs, while retaining the individual tracks."
        )

        self.export_button = QPushButton("Export combined video…")
        self.export_button.clicked.connect(self._choose_output)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_export)

        buttons = QHBoxLayout()
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addWidget(self.input_list)
        layout.addWidget(self.merged_audio_checkbox)
        layout.addLayout(buttons)
        layout.addWidget(self.result_label)
        layout.addStretch(1)
        self.refresh()

    def set_experiment(self, experiment: Experiment) -> None:
        # A newly opened experiment gets the promised all-checked initial state,
        # even when it happens to reuse ids from the previous one.
        self.input_list.clear()
        self.result_label.clear()
        super().set_experiment(experiment)

    def refresh(self) -> None:
        if self._thread is not None:
            return
        checked = {
            self.input_list.item(index).data(_INPUT_ID_ROLE): self.input_list.item(
                index
            ).checkState()
            == Qt.CheckState.Checked
            for index in range(self.input_list.count())
        }
        self.input_list.blockSignals(True)
        self.input_list.clear()
        for data in self.experiment.inputs:
            item = QListWidgetItem(f"{data.id} ({_input_kind(data)})")
            item.setData(_INPUT_ID_ROLE, data.id)
            item.setData(_IS_VIDEO_ROLE, isinstance(data, Video))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if checked.get(data.id, True)
                else Qt.CheckState.Unchecked
            )
            self.input_list.addItem(item)
        self.input_list.blockSignals(False)
        self._update_availability()

    def selected_input_ids(self) -> list[str]:
        return [
            item.data(_INPUT_ID_ROLE)
            for index in range(self.input_list.count())
            if (item := self.input_list.item(index)).checkState()
            == Qt.CheckState.Checked
        ]

    def is_busy(self) -> bool:
        return self._thread is not None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @Slot()
    def _update_availability(self) -> None:
        has_selected_video = any(
            self.input_list.item(index).checkState() == Qt.CheckState.Checked
            and bool(self.input_list.item(index).data(_IS_VIDEO_ROLE))
            for index in range(self.input_list.count())
        )
        running = self._thread is not None
        self.input_list.setEnabled(not running)
        self.merged_audio_checkbox.setEnabled(not running)
        self.export_button.setEnabled(not running and has_selected_video)
        self.cancel_button.setVisible(running)

    @Slot()
    def _choose_output(self) -> None:
        if self._thread is not None or not self.export_button.isEnabled():
            return
        folder = self.experiment.folder or Path.cwd()
        chosen, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export combined video",
            str(folder / "combined_video.mp4"),
            "MP4 video (*.mp4)",
        )
        if not chosen:
            return
        output_path = Path(chosen)
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
        self._start_export(output_path)

    def _start_export(self, output_path: Path) -> None:
        input_ids = self.selected_input_ids()
        if self._thread is not None or not input_ids:
            return
        self.result_label.setText(f"Exporting to {output_path}…")
        self._worker = _VideoExportWorker(
            self.experiment,
            output_path,
            input_ids,
            self.merged_audio_checkbox.isChecked(),
        )
        self._worker.progress.connect(
            lambda value: self.progress_changed.emit(
                value, 100, "Exporting combined video…"
            )
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self.progress_changed.emit(0, 100, "Exporting combined video…")
        self.busy_changed.emit(True)
        self._update_availability()
        self._thread.start()

    @Slot()
    def _cancel_export(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.result_label.setText("Cancelling export…")

    @Slot(object)
    def _on_finished(self, result: VideoGridResult) -> None:
        message = (
            f"Exported combined video to {result.path} and ELAN annotations to "
            f"{result.path.with_suffix('.eaf')}"
        )
        self.result_label.setText(message)
        self.status_message.emit(message)
        self._set_running(False)

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Video export failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self.result_label.setText("Could not export combined video.")
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self.result_label.setText("Combined video export cancelled.")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self._thread = None
            self._worker = None
            self.cancel_button.setEnabled(True)
        self.busy_changed.emit(running)
        self._update_availability()
