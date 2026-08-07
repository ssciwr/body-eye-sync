"""Temporary one-click automatic alignment tab."""

from __future__ import annotations

from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.prepare import align_experiment
from body_eye_sync.gui.tabs.base import BaseTab


class AlignmentTab(BaseTab):
    """Populate input offsets using automatic audio alignment."""

    title = "Alignment"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.align_button = QPushButton("Automatic alignment")
        self.align_button.clicked.connect(self._align)

        layout = QVBoxLayout(self)
        layout.addWidget(self.align_button)
        layout.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        self.align_button.setEnabled(len(self._inputs()) >= 2)

    def _align(self) -> None:
        if len(self._inputs()) < 2:
            return
        self.align_button.setEnabled(False)
        self.busy_changed.emit(True)
        self.progress_changed.emit(0, 100, "Aligning recordings…")
        try:
            result = align_experiment(self.experiment, progress=self._progress)
            if result.offsets:
                self.experiment_changed.emit()
                self.status_message.emit("Automatic alignment finished")
        finally:
            self.busy_changed.emit(False)
            self.refresh()

    def _progress(self, value: float) -> bool:
        self.progress_changed.emit(round(100 * value), 100, "Aligning recordings…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        return True
