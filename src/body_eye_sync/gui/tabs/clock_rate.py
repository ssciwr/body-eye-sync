"""Clock rate tab: measure each input's clock rate against the others."""

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
from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.experiment.preprocess import (
    apply_clock_rates,
    clear_clock_rates,
    has_corrected_clock_rates,
)
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets.auto_height_table import AutoHeightTable
from body_eye_sync.media import media_duration
from body_eye_sync.preprocessing.clock_rate import (
    DEFAULT_SEARCH,
    DEFAULT_WINDOW,
    MIN_DRIFT_PPM,
    SPECTRAL_MIN_QUALITY,
    ClockRateAnalysis,
    ClockRateAnalysisCancelled,
    analyse_clock_rates,
)

_ID, _OFFSET, _DRIFT = range(3)
_COLUMNS = ["Id", "Offset", "Clock drift"]

_LABEL = "Analysing clock rates…"


def _offset_text(offset: float) -> str:
    return f"{offset:+.3f} s"


def _drift_text(timeline: Timeline) -> str:
    """One recording's clock rate, as parts per million."""
    return "0" if not timeline.corrects_drift else f"{timeline.drift_ppm:+.1f} ppm"


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
    """Run one clock-rate analysis without blocking Qt's event loop."""

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
        percent = round(100 * value)
        if percent != self._reported:
            self._reported = percent
            self.progress.emit(percent)
        return not self._cancel.is_set()

    @Slot()
    def run(self) -> None:
        try:
            result = analyse_clock_rates(
                self._paths,
                self._offsets,
                progress=self._progress,
                **self._settings,
            )
        except ClockRateAnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit(result)


class ClockRateTab(BaseTab):
    """Recalculate and apply each input's offset and clock rate."""

    title = "Clock rate"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self._thread: threading.Thread | None = None
        self._worker: _Worker | None = None
        self._analysis: ClockRateAnalysis | None = None
        self._analysis_signature: tuple | None = None

        self.table = AutoHeightTable(_COLUMNS)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            _DRIFT, QHeaderView.ResizeMode.Stretch
        )

        self.correct_button = QPushButton("Analyse and correct clock rates")
        self.correct_button.clicked.connect(self._start_correction)
        self.clear_button = QPushButton("Clear corrections")
        self.clear_button.setToolTip(
            "Reset every input's clock rate, leaving the offsets alone"
        )
        self.clear_button.clicked.connect(self._clear_corrections)
        self.window_spin = _setting_spin(2.0, 120.0, DEFAULT_WINDOW)
        self.window_spin.setToolTip("How long each measurement window is.")
        self.search_spin = _setting_spin(1.0, 120.0, DEFAULT_SEARCH)
        self.search_spin.setToolTip(
            "How far either side of the current offset each window looks for its lag."
        )
        self.min_quality_spin = _setting_spin(1.0, 30.0, SPECTRAL_MIN_QUALITY)
        self.min_quality_spin.setToolTip(
            "Quality gate: higher values require a stronger signal to measure a lag."
        )
        self.min_drift_spin = _setting_spin(0.5, 50.0, MIN_DRIFT_PPM)
        self.min_drift_spin.setDecimals(1)
        self.min_drift_spin.setSingleStep(0.5)
        self.min_drift_spin.setToolTip(
            "The smallest clock difference worth correcting, in parts per million."
        )
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.addRow("Window (s)", self.window_spin)
        settings_form.addRow("Search (s)", self.search_spin)
        settings_form.addRow("Min quality", self.min_quality_spin)
        settings_form.addRow("Min drift (ppm)", self.min_drift_spin)
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
                data.timeline.offset,
                data.timeline.rate,
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
                _offset_text(data.timeline.offset),
                _drift_text(data.timeline),
            ]
            if data.id in unavailable:
                values[2] = "Couldn't match"
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == _OFFSET:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
        self.table.fit_to_rows()

    def is_busy(self) -> bool:
        return self._thread is not None

    def _clear_corrections(self) -> None:
        if self._thread is not None or not clear_clock_rates(self.experiment):
            return
        self._analysis = None
        self._analysis_signature = None
        self._refresh_table()
        self._draw_stored_corrections()
        self._update_buttons(False)
        self.experiment_changed.emit()
        self.status_message.emit("Clock-rate corrections cleared")

    def _start_correction(self) -> None:
        if self._thread is not None:
            return
        paths = {name: data.path for name, data in self._inputs().items()}
        if len(paths) < 2:
            return
        offsets = {name: data.timeline.offset for name, data in self._inputs().items()}
        self._analysis = None
        self._analysis_signature = None
        self._show_plot(False)
        self._set_running(True)
        self.progress_changed.emit(0, 100, _LABEL)
        self._worker = _Worker(paths, offsets, self._settings())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_correction_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    @Slot(int)
    def _on_progress(self, percent: int) -> None:
        self.progress_changed.emit(percent, 100, _LABEL)

    @Slot(object)
    def _on_correction_finished(self, analysis: ClockRateAnalysis) -> None:
        changed = apply_clock_rates(self.experiment, analysis)
        self._analysis = analysis
        self._analysis_signature = self._timeline_signature()
        self._refresh_table()

        self._draw_corrections(
            {name: data.timeline for name, data in self._inputs().items()}, analysis
        )
        self._show_plot(True)
        if changed:
            self.experiment_changed.emit()
            self.status_message.emit(
                f"Updated the clock rate of {len(changed)} input(s)"
            )
        else:
            self.status_message.emit("No clock-rate changes detected")
        self._set_running(False)

    def _draw_stored_corrections(self) -> None:
        timelines = {name: data.timeline for name, data in self._inputs().items()}
        if not any(timeline.corrects_drift for timeline in timelines.values()):
            self._show_plot(False)
            return
        self._draw_corrections(timelines)
        self._show_plot(True)

    def _draw_corrections(
        self,
        timelines: dict[str, Timeline],
        analysis: ClockRateAnalysis | None = None,
    ) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        inputs = self._inputs()
        for index, (name, timeline) in enumerate(timelines.items()):
            if name not in inputs:
                continue
            points = analysis.points.get(name, []) if analysis is not None else []
            if not points and not timeline.corrects_drift:
                continue
            if points:
                measured_experiment = np.asarray([point.time for point in points])
                measured_offset = np.asarray([point.offset for point in points])
                local = measured_experiment - measured_offset
            else:
                duration = media_duration(inputs[name].path) or 3600.0
                local = np.asarray([0.0, duration])
            fitted = timeline.to_experiment_times(local)
            experiment = measured_experiment if points else fitted
            fitted_offset = fitted - local
            colour = f"C{index}"
            if points:
                axis.scatter(
                    experiment / 60.0,
                    (measured_offset - timeline.offset) * 1000,
                    s=9,
                    alpha=0.45,
                    color=colour,
                )
            axis.plot(
                experiment / 60.0,
                (fitted_offset - timeline.offset) * 1000,
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
                else "No clock drift"
            ),
        )
        corrected = any(
            timeline.corrects_drift
            for name, timeline in timelines.items()
            if name in inputs
        )
        if analysis is None:
            title = "Applied clock-rate corrections"
        elif corrected:
            title = "Measured offsets and applied clock-rate corrections"
        else:
            title = "Measured offsets: no clock-rate correction needed"
        axis.set_title(title)
        axis.set_xlabel("Experiment time (minutes)")
        axis.set_ylabel("Offset change from recording start (ms)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        self.canvas.draw_idle()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Clock-rate analysis failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self.status_message.emit("Could not complete the clock-rate analysis")
        self._draw_stored_corrections()
        self._set_running(False)

    @Slot()
    def _on_cancelled(self) -> None:
        self.status_message.emit("Clock-rate analysis cancelled")
        self._draw_stored_corrections()
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        if not running:
            self._thread = None
            self._worker = None
        self._update_buttons(running)
        self.busy_changed.emit(running)

    def _settings(self) -> dict[str, float]:
        """The analysis settings as the form currently has them."""
        return {
            "window": self.window_spin.value(),
            "search": self.search_spin.value(),
            "min_quality": self.min_quality_spin.value(),
            "min_drift_ppm": self.min_drift_spin.value(),
        }

    def _update_buttons(self, running: bool) -> None:
        self.settings_widget.setEnabled(not running)
        self.correct_button.setEnabled(not running and len(self._inputs()) >= 2)
        self.clear_button.setEnabled(
            not running and has_corrected_clock_rates(self.experiment)
        )
