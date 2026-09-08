"""Common base class for tabs.

The main window owns the experiment, and passes the experiment to a tab using
:meth:`BaseTab.set_experiment`, and calls :meth:`BaseTab.refresh` when the experiment is changed elsewhere.
Tabs report changes to the experiment via signals.
"""

from __future__ import annotations

import threading
from typing import ClassVar

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video

SHUTDOWN_TIMEOUT = 5.0


class BaseTab(QWidget):
    """One tab of the main window, acting on an :class:`Experiment`."""

    title: ClassVar[str] = ""

    _worker: QObject | None = None
    _thread: threading.Thread | None = None

    # signals
    status_message = Signal(str)
    experiment_changed = Signal()
    busy_changed = Signal(bool)
    finished = Signal()
    # current value, maximum value (zero means indeterminate), operation label:
    progress_changed = Signal(int, int, str)

    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self.experiment = experiment

    def set_experiment(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.refresh()

    def _inputs(self) -> dict[str, Video | Audio]:
        """The experiment's inputs that have a recording, keyed by id."""
        return {
            data.id: data for data in self.experiment.inputs if data.path is not None
        }

    def refresh(self) -> None:
        """Re-read the experiment, which may have been changed elsewhere."""

    def shutdown(self) -> None:
        """Stop any work in progress; called when the window is closing."""
        worker, thread = self._worker, self._thread
        self._worker = None
        self._thread = None
        if worker is None:
            return
        worker.cancel()
        QObject.disconnect(worker, None, None, None)
        if thread is not None:
            thread.join(timeout=SHUTDOWN_TIMEOUT)


class PlaceholderTab(BaseTab):
    """A temporary tab for yet to be implemented tabs."""

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        label = QLabel(f"{self.title} is not implemented yet")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
