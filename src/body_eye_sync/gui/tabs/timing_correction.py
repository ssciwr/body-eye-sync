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
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
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
    clear_timing_corrections,
    has_timing_corrections,
    stored_fits,
)
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets.auto_height_table import AutoHeightTable
from body_eye_sync.media import media_duration
from body_eye_sync.preprocessing.alignment import (
    DEFAULT_MIN_SHIFT,
    SPECTRAL,
    Shift,
    TimelineFit,
    to_experiment_times,
)
from body_eye_sync.preprocessing.timing_correction import (
    DEFAULT_SEARCH,
    DEFAULT_WINDOW,
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


def _setting_spin(low: float, high: float, value: float) -> QDoubleSpinBox:
    """A spin box for one analysis setting, bounded to values worth trying."""
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(0.5 if high <= 30.0 else 1.0)
    spin.setDecimals(1)
    spin.setValue(value)
    spin.setMaximumWidth(90)
    return spin


class _Worker(QObject):
    """Run one timing task without blocking Qt's event loop."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(
        self,
        paths: dict[str, Path],
        offsets: dict[str, float],
        settings: dict[str, float],
    ) -> None:
        super().__init__()
        self._paths = paths
        self._offsets = offsets
        self._settings = settings
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
                **self._settings,
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
        self.table.horizontalHeader().setSectionResizeMode(
            _GAPS, QHeaderView.ResizeMode.Stretch
        )

        self.correct_button = QPushButton("Analyse and correct timing")
        self.correct_button.clicked.connect(self._start_correction)
        self.clear_button = QPushButton("Clear corrections")
        self.clear_button.setToolTip(
            "Reset every input's drift and gaps, leaving the offsets alone"
        )
        self.clear_button.clicked.connect(self._clear_corrections)
        self.window_spin = _setting_spin(2.0, 120.0, DEFAULT_WINDOW)
        self.window_spin.setToolTip(
            "How long each measurement window is. Shorter places a gap more "
            "precisely; longer locks onto quiet or sparse recordings more often."
        )
        self.search_spin = _setting_spin(1.0, 120.0, DEFAULT_SEARCH)
        self.search_spin.setToolTip(
            "How far either side of the current offset each window looks for "
            "its lag. Widen it when the inputs are only roughly aligned."
        )
        self.min_quality_spin = _setting_spin(1.0, 30.0, SPECTRAL.min_quality)
        self.min_quality_spin.setToolTip(
            "How far a window's best lag must stand above the rest before it is "
            "believed. Lower it to measure difficult recordings, at the risk of "
            "reading a wrong lag as a gap."
        )
        self.min_gap_spin = _setting_spin(10.0, 500.0, 1000 * DEFAULT_MIN_SHIFT)
        self.min_gap_spin.setDecimals(0)
        self.min_gap_spin.setSingleStep(10.0)
        self.min_gap_spin.setToolTip(
            "The smallest step in the offset worth calling lost content. "
            "Lower it to find more, smaller gaps, at the risk of reading "
            "measurement noise as one."
        )
        self.fit_drift_check = QCheckBox()
        self.fit_drift_check.setToolTip(
            "Let this recording have a clock rate of its own. Off by default: "
            "a free rate absorbs a run of small gaps into a slope no crystal "
            "could produce. Turn it on when the offset really does climb "
            "steadily between the steps."
        )

        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.addRow("Window (s)", self.window_spin)
        settings_form.addRow("Search (s)", self.search_spin)
        settings_form.addRow("Min quality", self.min_quality_spin)
        settings_form.addRow("Min gap (ms)", self.min_gap_spin)
        settings_form.addRow("Fit drift", self.fit_drift_check)
        self.settings_widget = QWidget()
        self.settings_widget.setLayout(settings_form)

        self.figure = Figure(figsize=(9, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(400)

        buttons = QHBoxLayout()
        buttons.addWidget(self.correct_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)

        controls = QVBoxLayout()
        controls.addWidget(self.settings_widget)
        controls.addLayout(buttons)
        controls.addStretch(1)

        top = QHBoxLayout()
        top.addWidget(self.table, 1, Qt.AlignmentFlag.AlignTop)
        top.addLayout(controls)

        page_layout = QVBoxLayout()
        page_layout.addLayout(top)
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
        self._refresh_table()
        if self._analysis is None:
            self._draw_stored_corrections()
        self._update_buttons(False)

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

    def _clear_corrections(self) -> None:
        """Put every input back on its own unstretched, ungapped clock."""
        if self._thread is not None or not clear_timing_corrections(self.experiment):
            return
        self._analysis = None
        self._analysis_signature = None
        self._refresh_table()
        self._draw_stored_corrections()
        self._update_buttons(False)
        self.experiment_changed.emit()
        self.status_message.emit("Timing corrections cleared")

    def _start_correction(self) -> None:
        if self._thread is not None:
            return
        paths = {name: data.path for name, data in self._inputs().items()}
        if len(paths) < 2:
            return
        offsets = {name: data.time_offset for name, data in self._inputs().items()}
        self._analysis = None
        self._analysis_signature = None
        self._show_plot(False)
        self._set_running(True)
        self.progress_changed.emit(0, 100, "Analysing timing…")
        self._worker = _Worker(paths, offsets, self._settings())
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
            self._draw_corrections(analysis.fits, analysis)
            self._show_plot(True)
            self.experiment_changed.emit()
            self.status_message.emit(
                f"Corrected drift or gaps in {len(corrected)} input(s)"
            )
        else:
            self._draw_stored_corrections()
            self.status_message.emit("No drift or gaps detected")
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
        self.status_message.emit("Could not complete timing correction")
        self._draw_stored_corrections()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self.status_message.emit("Timing correction cancelled")
        self._draw_stored_corrections()
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self._thread = None
            self._worker = None
        self._update_buttons(running)
        self.busy_changed.emit(running)

    def _settings(self) -> dict[str, float]:
        """The analysis settings as the form currently has them.

        ``step`` is left out on purpose: the windows are spaced at half a window,
        which is the spacing the gap fit is calibrated for.
        """
        return {
            "window": self.window_spin.value(),
            "search": self.search_spin.value(),
            "min_quality": self.min_quality_spin.value(),
            "min_shift": self.min_gap_spin.value() / 1000.0,
            "fit_drift": self.fit_drift_check.isChecked(),
        }

    def _update_buttons(self, running: bool) -> None:
        """Correcting needs something to compare; clearing needs something to undo."""
        self.settings_widget.setEnabled(not running)
        self.correct_button.setEnabled(not running and len(self._inputs()) >= 2)
        self.clear_button.setEnabled(
            not running and has_timing_corrections(self.experiment)
        )
