"""Alignment tab: placing each input on the shared experiment timeline."""

from __future__ import annotations

from qtpy.QtCore import Signal
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
_VIDEOS_PER_ROW = 3
_SET_BUTTON_ACTIVE_STYLE = (
    "QToolButton { background-color: #2563eb; color: white; font-weight: 600; }"
)
_THIS_VIDEO_BUTTON_STYLE = (
    "QPushButton { background-color: #16a34a; color: white; font-weight: 600; }"
)
_ALL_VIDEOS_BUTTON_STYLE = (
    "QPushButton { background-color: #dc2626; color: white; font-weight: 600; }"
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


class _VideoAlignmentCard(QWidget):
    """One video viewer and its offset editor."""

    changed = Signal()
    set_requested = Signal(object)

    def __init__(self, video: Video) -> None:
        super().__init__()
        self.video = video
        self.load_error: OSError | None = None
        self.viewer = VideoViewer()
        self.viewer.show_overlays = False
        # This is here to clear viewer issues when .load goes wrong for some reason.
        try:
            self.viewer.load(video)
        except OSError as exc:
            self.load_error = exc
            self.viewer.clear()

        self.controls = _VideoAlignmentControls(video, self.viewer)
        if self.load_error is not None:
            self.setEnabled(False)
        self.controls.spin.valueChanged.connect(lambda _value: self.changed.emit())
        self.controls.set_button.clicked.connect(
            lambda _checked=False: self.set_requested.emit(self)
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.viewer)
        layout.addWidget(self.controls)

    @property
    def loaded(self) -> bool:
        return self.load_error is None

    def set_offset(self, offset: float) -> None:
        self.controls.spin.setValue(offset)

    def set_offset_from_current_time(self) -> None:
        self.set_offset(round(-self.viewer.current_time_seconds, 3))

    def shutdown(self) -> None:
        self.viewer.clear()


class AlignmentTab(BaseTab):
    """Let the user manually align all kind of inputs in time with each other via setting their time_offset properties."""

    title = "Alignment"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.video_cards: list[_VideoAlignmentCard] = []
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
        for card in self.video_cards:
            self.grid.removeWidget(card)
            card.shutdown()
            card.deleteLater()
        self.video_cards = []

        videos = [*self.experiment.glasses_videos, *self.experiment.fixed_videos]
        for index, video in enumerate(videos):
            card = _VideoAlignmentCard(video)
            if card.load_error is not None:
                self.status_message.emit(f"Could not open video: {card.load_error}")
            card.changed.connect(self.experiment_changed)
            card.set_requested.connect(self._set_offset_from_current_frame)

            self.video_cards.append(card)
            self.grid.addWidget(card, index // _VIDEOS_PER_ROW, index % _VIDEOS_PER_ROW)

    def _set_offset_from_current_frame(self, source: _VideoAlignmentCard) -> None:
        source.set_offset_from_current_time()
        loaded_cards = [card for card in self.video_cards if card.loaded]
        message = QMessageBox(self)
        message.setWindowTitle("Set offset")
        message.setText(
            "Do you want to set the offset for only this video, or for all videos?"
        )
        message.setIcon(QMessageBox.Icon.Question)
        this_video_button = message.addButton(
            "This video", QMessageBox.ButtonRole.AcceptRole
        )
        all_videos_button = message.addButton(
            "All videos", QMessageBox.ButtonRole.DestructiveRole
        )
        this_video_button.setStyleSheet(_THIS_VIDEO_BUTTON_STYLE)
        all_videos_button.setStyleSheet(_ALL_VIDEOS_BUTTON_STYLE)
        message.setDefaultButton(this_video_button)
        message.exec()
        if message.clickedButton() is not all_videos_button:
            return
        for card in loaded_cards:
            if card is not source:
                card.set_offset(source.video.time_offset)
