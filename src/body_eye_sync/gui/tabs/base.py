"""Common base class for tabs.

The main window owns the experiment, and passes the experiment to a tab using
:meth:`BaseTab.set_experiment`, and calls :meth:`BaseTab.refresh` when the experiment is changed elsewhere.
Tabs report changes to the experiment via signals.
"""

from __future__ import annotations

from typing import ClassVar

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from body_eye_sync.experiment.experiment import Experiment


class BaseTab(QWidget):
    """One tab of the main window, acting on an :class:`Experiment`."""

    title: ClassVar[str] = ""

    # signals
    status_message = Signal(str)
    experiment_changed = Signal()
    busy_changed = Signal(bool)
    finished = Signal()

    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self.experiment = experiment

    def set_experiment(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.refresh()

    def refresh(self) -> None:
        """Re-read the experiment, which may have been changed elsewhere."""

    def shutdown(self) -> None:
        """Stop any work in progress; called when the window is closing."""


class PlaceholderTab(BaseTab):
    """A temporary tab for yet to be implemented tabs."""

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        label = QLabel(f"{self.title} is not implemented yet")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
