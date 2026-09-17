"""Cluster glasses-video tracklets and identify their participants."""

from __future__ import annotations

import threading
import traceback

import pandas as pd
from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.experiment.config import ClusterPostProcessingSettings
from body_eye_sync.experiment.postprocess import (
    cluster_experiment_tracklets,
    clustering_blocked_reason,
)
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.utils import recording_colors
from body_eye_sync.gui.widgets import PydanticForm, VideoSelectionWidget

_LABEL = "Clustering tracklets…"


class _Worker(QObject):
    finished = Signal()
    failed = Signal(str, str)

    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self._experiment = experiment
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Suppress completion signals when the tab shuts down."""
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            cluster_experiment_tracklets(self._experiment)
        except Exception as exc:
            if not self._cancelled.is_set():
                self.failed.emit(str(exc), traceback.format_exc())
        else:
            if not self._cancelled.is_set():
                self.finished.emit()


class ParticipantIdentificationTab(BaseTab):
    """Run clustering and view videos with identified glasses IDs."""

    title = "Participant identification"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.cluster_button = QPushButton("Identify participants")
        self.cluster_button.clicked.connect(self._start)
        self.blocked_label = QLabel()
        self.blocked_label.setWordWrap(True)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.video_panel = VideoSelectionWidget()
        self.video_panel.video_selected.connect(self._on_video_selected)
        self.video_panel.status_message.connect(self.status_message)
        self.video_selector = self.video_panel.selector
        self.video_viewer = self.video_panel.viewer
        results_side = QWidget()
        results_layout = QVBoxLayout(results_side)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addWidget(self.video_panel, stretch=1)
        results_layout.addWidget(self.summary_label)

        self.settings_form = PydanticForm(
            self.experiment.pipeline.cluster_post_processing
        )
        self.settings_form.changed.connect(self._on_settings_changed)
        self.settings_group = QGroupBox("Clustering settings")
        QVBoxLayout(self.settings_group).addWidget(self.settings_form)
        settings_side = QWidget()
        settings_layout = QVBoxLayout(settings_side)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(self.settings_group)
        settings_layout.addWidget(self.blocked_label)
        settings_layout.addWidget(self.cluster_button)
        settings_layout.addStretch(1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(results_side)
        self.splitter.addWidget(settings_side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        QVBoxLayout(self).addWidget(self.splitter)
        self.refresh()

    def blocked_reason(self) -> str | None:
        return clustering_blocked_reason(self.experiment)

    def is_busy(self) -> bool:
        return self._thread is not None

    def refresh(self) -> None:
        if self.is_busy():
            return
        self.settings_group.setEnabled(True)
        self.settings_form.blockSignals(True)
        self.settings_form.from_model(self.experiment.pipeline.cluster_post_processing)
        self.settings_form.blockSignals(False)
        blocked = self.blocked_reason()
        self.cluster_button.setEnabled(blocked is None)
        self.blocked_label.setText(blocked or "")
        self.blocked_label.setVisible(blocked is not None)
        data = self.experiment.identities.data
        if data is None:
            self.summary_label.setText("Clustering has not been run.")
        else:
            identified = int(data["participant_id"].notna().sum())
            self.summary_label.setText(
                f"{identified} of {len(data)} tracklets identified across "
                f"{len(self.experiment.identities.participants)} participants."
            )
        self.video_panel.set_videos(self.experiment.glasses_videos)

    @Slot()
    def _on_settings_changed(self) -> None:
        settings = self.settings_form.to_model()
        if isinstance(settings, ClusterPostProcessingSettings):
            self.experiment.pipeline.cluster_post_processing = settings
            self.experiment_changed.emit()

    def _on_video_selected(self, video: Video | None) -> None:
        data = (
            self.experiment.identities.for_video(video.id) if video else pd.DataFrame()
        )
        self.video_viewer.track_labels = {
            row.track_id: "Unidentified"
            if pd.isna(row.participant_id)
            else str(row.participant_id)
            for row in data.itertuples(index=False)
        }
        colors = recording_colors(data.id for data in self.experiment.inputs)
        self.video_viewer.track_colors = {
            row.track_id: colors[row.participant_id]
            for row in data.itertuples(index=False)
            if pd.notna(row.participant_id)
        }

    def _start(self) -> None:
        if self.is_busy() or self.blocked_reason() is not None:
            return
        self.cluster_button.setEnabled(False)
        self.settings_group.setEnabled(False)
        self.summary_label.setText(_LABEL)
        self._worker = _Worker(self.experiment)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self.busy_changed.emit(True)
        self.progress_changed.emit(0, 0, _LABEL)
        self._thread.start()

    @Slot()
    def _on_finished(self) -> None:
        self._finish()
        self.status_message.emit("Clustering completed.")
        self.experiment_changed.emit()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        self._finish()
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Clustering failed")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    def _finish(self) -> None:
        self._thread = None
        self._worker = None
        self.refresh()
        self.busy_changed.emit(False)
