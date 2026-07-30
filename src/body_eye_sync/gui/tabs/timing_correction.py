"""Timing correction tab: correct drift and gaps on every input timeline."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.prepare import (
    apply_timing_corrections,
    stored_fits,
)
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets.auto_height_table import AutoHeightTable
from body_eye_sync.media import media_duration
from body_eye_sync.preprocessing.alignment import (
    Shift,
    TimelineFit,
    to_experiment_times,
)
from body_eye_sync.preprocessing.timing_correction import (
    TimingCorrectionAnalysis,
    TimingCorrectionCancelled,
    analyse_timing_corrections,
)

_ID, _OFFSET, _DRIFT, _GAPS = range(4)
_COLUMNS = ["Id", "Offset", "Drift", "Gaps"]


def _offset_text(offset: float) -> str:
    return f"{offset:+.3f} s"


def _drift_text(scale: float) -> str:
    ppm = (scale - 1.0) * 1e6
    return "0 ppm" if abs(ppm) < 0.5 else f"{ppm:+.0f} ppm"


def _gaps_text(shifts: list[Shift]) -> str:
    return (
        ", ".join(f"{shift.at:.1f}s: {shift.seconds * 1000:.0f}ms" for shift in shifts)
        or "0"
    )


def _line_times(duration: float, shifts: list[Shift]) -> np.ndarray:
    """Local times spanning a recording, with points either side of each gap."""
    epsilon = max(1e-6, duration * 1e-9)
    times = [0.0, duration]
    for shift in shifts:
        if 0.0 < shift.at < duration:
            times.extend([max(0.0, shift.at - epsilon), shift.at])
    return np.asarray(sorted(set(times)))


class _Worker(QObject):
    """Run one timing task without blocking Qt's event loop."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, paths: dict[str, Path], offsets: dict[str, float]) -> None:
        super().__init__()
        self._paths = paths
        self._offsets = offsets
        self._cancel = threading.Event()
        self._reported = -1

    def cancel(self) -> None:
        self._cancel.set()

    def _progress(self, value: float) -> bool:
        # Called once per measurement window -- thousands of times for a long
        # experiment -- but the bar only has whole percent to show.
        percent = round(100 * value)
        if percent != self._reported:
            self._reported = percent
            self.progress.emit(percent)
        return not self._cancel.is_set()

    @Slot()
    def run(self) -> None:
        try:
            result = analyse_timing_corrections(
                self._paths,
                self._offsets,
                progress=self._progress,
            )
        except TimingCorrectionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit(result)


class TimingCorrectionTab(BaseTab):
    """Recalculate and apply offsets, drift and recording gaps."""

    title = "Timing correction"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self._thread: threading.Thread | None = None
        self._worker: _Worker | None = None
        self._analysis: TimingCorrectionAnalysis | None = None
        self._analysis_signature: tuple | None = None

        self.table = AutoHeightTable(_COLUMNS)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.correct_button = QPushButton("Analyse and correct timing")
        self.correct_button.clicked.connect(self._start_correction)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.figure = Figure(figsize=(9, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(400)

        page_layout = QVBoxLayout()
        page_layout.addWidget(self.correct_button)
        page_layout.addWidget(self.table)
        page_layout.addWidget(self.summary_label)
        page_layout.addWidget(self.canvas)
        page_layout.addStretch(1)

        self.page = QWidget()
        self.page.setLayout(page_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.page)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        self.refresh()

    def _show_plot(self, visible: bool) -> None:
        self.canvas.setVisible(visible)

    def _timeline_signature(self) -> tuple:
        return tuple(
            (
                name,
                str(data.path),
                data.time_offset,
                data.time_scale,
                tuple((shift.at, shift.seconds) for shift in data.time_shifts),
            )
            for name, data in self._inputs().items()
        )

    def set_experiment(self, experiment: Experiment) -> None:
        self._analysis = None
        self._analysis_signature = None
        self.summary_label.clear()
        self._show_plot(False)
        super().set_experiment(experiment)

    def refresh(self) -> None:
        if self._thread is not None:
            return
        if (
            self._analysis is not None
            and self._analysis_signature != self._timeline_signature()
        ):
            self._analysis = None
            self._analysis_signature = None
            self.summary_label.clear()
        self._refresh_table()
        if self._analysis is None:
            self._draw_stored_corrections()
        self.correct_button.setEnabled(len(self._inputs()) >= 2)

    def _refresh_table(self) -> None:
        inputs = list(self.experiment.inputs)
        unavailable = set(self._analysis.unavailable) if self._analysis else set()
        self.table.clearContents()
        self.table.setRowCount(len(inputs))
        for row, data in enumerate(inputs):
            values = [
                data.id,
                _offset_text(data.time_offset),
                _drift_text(data.time_scale),
                _gaps_text(data.time_shifts),
            ]
            # A drift of "0 ppm" and no gaps would claim this recording kept
            # time; not placing it against the reference is a different thing.
            if data.id in unavailable:
                values[2:] = ["Couldn't match", "Couldn't match"]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (_OFFSET, _DRIFT):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
        self.table.fit_to_rows()

    def is_busy(self) -> bool:
        return self._thread is not None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _start_correction(self) -> None:
        if self._thread is not None:
            return
        paths = {name: data.path for name, data in self._inputs().items()}
        if len(paths) < 2:
            return
        offsets = {name: data.time_offset for name, data in self._inputs().items()}
        self._analysis = None
        self._analysis_signature = None
        self.summary_label.setText("Measuring drift and gaps…")
        self._show_plot(False)
        self._set_running(True)
        self.progress_changed.emit(0, 100, "Analysing timing…")
        self._worker = _Worker(paths, offsets)
        self._worker.progress.connect(
            lambda value: self.progress_changed.emit(value, 100, "Analysing timing…")
        )
        self._worker.finished.connect(self._on_correction_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    @Slot(object)
    def _on_correction_finished(self, analysis: TimingCorrectionAnalysis) -> None:
        corrected = apply_timing_corrections(self.experiment, analysis)
        self._analysis = analysis
        self._analysis_signature = self._timeline_signature()
        self._refresh_table()

        if corrected:
            self.summary_label.setText(
                f"Corrected drift or gaps in {len(corrected)} input(s)."
            )
            self._draw_corrections(analysis.fits, analysis)
            self._show_plot(True)
            self.experiment_changed.emit()
            self.status_message.emit("Timing corrections applied")
        else:
            self.summary_label.setText("No drift or gaps detected")
            self._draw_stored_corrections()
        self._set_running(False)

    def _draw_stored_corrections(self) -> None:
        fits = stored_fits(self.experiment)
        if not any(fit.corrects_timing for fit in fits.values()):
            self._show_plot(False)
            return
        self._draw_corrections(fits)
        self._show_plot(True)

    def _draw_corrections(
        self,
        fits: dict[str, TimelineFit],
        analysis: TimingCorrectionAnalysis | None = None,
    ) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        inputs = self._inputs()
        for index, (name, fit) in enumerate(fits.items()):
            if not fit.corrects_timing or name not in inputs:
                continue
            points = analysis.points.get(name, []) if analysis is not None else []
            if points:
                measured_experiment = np.asarray([point.time for point in points])
                measured_offset = np.asarray([point.offset for point in points])
                local = measured_experiment - measured_offset
            else:
                duration = media_duration(inputs[name].path)
                if duration is None:
                    duration = (
                        max((shift.at for shift in fit.shifts), default=3600.0) + 60.0
                    )
                local = _line_times(duration, fit.shifts)
            # Where the fit puts each of those local times, and so what it says
            # the offset is there: the curve drawn through the measurements.
            fitted = to_experiment_times(local, fit.offset, fit.shifts, fit.scale)
            experiment = measured_experiment if points else fitted
            fitted_offset = fitted - local
            colour = f"C{index}"
            if points:
                axis.scatter(
                    experiment / 60.0,
                    (measured_offset - fit.offset) * 1000,
                    s=9,
                    alpha=0.45,
                    color=colour,
                )
            axis.plot(
                experiment / 60.0,
                (fitted_offset - fit.offset) * 1000,
                color=colour,
                linewidth=1.5,
                label=name,
            )
        axis.axhline(
            0,
            color="black",
            linewidth=0.8,
            label=(
                f"{analysis.reference} (reference)"
                if analysis is not None
                else "No timing correction"
            ),
        )
        axis.set_title("Applied timeline corrections")
        axis.set_xlabel("Experiment time (minutes)")
        axis.set_ylabel("Offset change from recording start (ms)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        self.canvas.draw_idle()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Timing correction failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self.summary_label.setText("Could not complete timing correction.")
        self._draw_stored_corrections()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self.summary_label.setText("Timing correction cancelled.")
        self._draw_stored_corrections()
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self._thread = None
            self._worker = None
        self.correct_button.setEnabled(not running and len(self._inputs()) >= 2)
        self.busy_changed.emit(running)
