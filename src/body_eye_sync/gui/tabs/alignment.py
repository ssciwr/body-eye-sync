"""Alignment tab: placing each input on the shared experiment timeline."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import VideoViewer

_OFFSET_STEP = 0.05
_SET_BUTTON_ACTIVE_STYLE = (
    "QToolButton { background-color: #2563eb; color: white; font-weight: 600; }"
)


class _VideoAlignmentControls(QWidget):
    def __init__(self, video: Video, viewer: VideoViewer) -> None:
        super().__init__()
        self.video = video
        self.viewer = viewer

        self.down_button = QToolButton()
        self.down_button.setText("-")
        self.down_button.setToolTip("Decrease offset by 0.05 s")
        self.up_button = QToolButton()
        self.up_button.setText("+")
        self.up_button.setToolTip("Increase offset by 0.05 s")
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-86_400.0, 86_400.0)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(_OFFSET_STEP)
        self.spin.setSuffix(" s")
        self.spin.setKeyboardTracking(False)
        self.spin.setMaximumWidth(105)
        self.spin.setValue(video.time_offset)
        self.set_button = QToolButton()
        self.set_button.setText("Set")
        self.set_button.setToolTip("Set as offset")
        self.time_label = QLabel()
        self.time_label.setToolTip("Video time -> timeline time")

        self.down_button.clicked.connect(
            lambda _checked=False: self.spin.setValue(self.spin.value() - _OFFSET_STEP)
        )
        self.up_button.clicked.connect(
            lambda _checked=False: self.spin.setValue(self.spin.value() + _OFFSET_STEP)
        )
        self.spin.valueChanged.connect(self._offset_changed)
        self.viewer.frame_changed.connect(self._refresh_time_label)
        self._refresh_time_label()

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel("Offset"))
        layout.addWidget(self.down_button)
        layout.addWidget(self.spin)
        layout.addWidget(self.up_button)
        layout.addWidget(self.set_button)
        layout.addWidget(self.time_label, stretch=1)

    def _offset_changed(self, value: float) -> None:
        offset = round(value, 3)
        if self.video.time_offset == offset:
            return
        self.video.time_offset = offset
        self.viewer.set_time_seconds(-offset, allow_negative=True)
        self._refresh_time_label()

    def _refresh_time_label(self, _frame: int = 0) -> None:
        video_time = self.viewer.current_time_seconds
        timeline_time = video_time + self.video.time_offset
        self.time_label.setText(f"{video_time:.3f} -> {timeline_time:.3f} s")
        active = round(timeline_time, 3) != 0.0
        self.set_button.setProperty("needsOffset", active)
        self.set_button.setStyleSheet(_SET_BUTTON_ACTIVE_STYLE if active else "")


class AlignmentTab(BaseTab):
    """Let the user manually align all kind of inputs in time with each other via setting their time_offset properties."""

    title = "Alignment"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self._cells: list[QWidget] = []
        self.video_viewers: list[VideoViewer] = []
        self.video_controls: list[_VideoAlignmentControls] = []
        self.done_button = QPushButton("Finish alignment")
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(self.finished.emit)

        layout = QVBoxLayout(self)
        self.grid = QGridLayout()
        layout.addLayout(self.grid, stretch=1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        """Render every video input, with at most three viewers per row."""
        for widget in self._cells:
            self.grid.removeWidget(widget)
            widget.deleteLater()
        self._cells = []
        self.video_viewers = []
        self.video_controls = []

        videos = [*self.experiment.glasses_videos, *self.experiment.fixed_videos]
        for index, video in enumerate(videos):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            viewer = VideoViewer()
            viewer.show_overlays = False
            try:
                viewer.load(video)
            except OSError as exc:
                self.status_message.emit(f"Could not open video: {exc}")

            controls = _VideoAlignmentControls(video, viewer)
            controls.spin.valueChanged.connect(
                lambda _value: self.experiment_changed.emit()
            )
            controls.set_button.clicked.connect(
                lambda _checked=False, c=controls: self._set_offset_from_current_frame(
                    c
                )
            )

            cell_layout.addWidget(viewer)
            cell_layout.addWidget(controls)
            self._cells.append(cell)
            self.video_viewers.append(viewer)
            self.video_controls.append(controls)
            self.grid.addWidget(cell, index // 3, index % 3)

    def _set_offset_from_current_frame(self, source: _VideoAlignmentControls) -> None:
        source.spin.setValue(round(-source.viewer.current_time_seconds, 3))
        if len(self.video_controls) <= 1:
            return
        answer = QMessageBox.question(
            self,
            "Set other videos?",
            "Set the other videos to this same video timestamp?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for controls in self.video_controls:
            if controls is not source:
                controls.spin.setValue(source.video.time_offset)
