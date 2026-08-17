"""Alignment tab: placing each input on the shared experiment timeline."""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
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
    "QToolButton { background-color: #2563eb; color: white; font-weight: 600; }"  # blue
)
_TIMELINE_LABEL_STYLE = "color: #9ca3af;"  # grey
_PRE_SHARED_LABEL_STYLE = "color: #dc2626; font-weight: 600;"  # red
_THIS_VIDEO_BUTTON_STYLE = "QPushButton { background-color: #16a34a; color: white; font-weight: 600; }"  # green
_ALL_VIDEOS_BUTTON_STYLE = (
    "QPushButton { background-color: #dc2626; color: white; font-weight: 600; }"  # red
)


class _VideoAlignmentControls(QWidget):
    timeline_changed = Signal(float)

    def __init__(self, video: Video, viewer: VideoViewer) -> None:
        super().__init__()
        self.video = video
        self.viewer = viewer
        self._preserve_timeline_on_offset_change = True

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
        self.set_button.setText("Zero to frame")
        self.set_button.setToolTip("previewed frame above")

        self.down_button.clicked.connect(
            lambda _checked=False: self.spin.setValue(self.spin.value() - _OFFSET_STEP)
        )
        self.up_button.clicked.connect(
            lambda _checked=False: self.spin.setValue(self.spin.value() + _OFFSET_STEP)
        )
        self.spin.valueChanged.connect(self._offset_changed)
        self.spin.lineEdit().textEdited.connect(self._refresh_timeline_state)
        self.viewer.frame_changed.connect(self._refresh_timeline_state)
        self._refresh_timeline_state()

        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setSpacing(4)
        layout.addWidget(self.down_button)
        layout.addWidget(self.spin)
        layout.addWidget(self.up_button)
        layout.addWidget(self.set_button)

    def set_offset(self, offset: float, *, preserve_timeline: bool = True) -> None:
        self._preserve_timeline_on_offset_change = preserve_timeline
        self.spin.setValue(offset)
        self._preserve_timeline_on_offset_change = True

    def _offset_changed(self, value: float) -> None:
        offset = round(value, 3)
        if self.video.time_offset == offset:
            return
        shared_timeline_time = (
            self.viewer.current_time_seconds + self.video.time_offset
            if self._preserve_timeline_on_offset_change
            else 0.0
        )
        self.video.time_offset = offset
        video_time = shared_timeline_time - offset
        self.viewer.set_time_seconds(
            video_time, allow_negative=True, show_requested_time=True
        )
        self._show_timeline_state(shared_timeline_time)

    def _refresh_timeline_state(self, _frame: object = None) -> None:
        self._show_timeline_state(self.viewer.current_time_seconds + self.offset())

    def _show_timeline_state(self, shared_timeline_time: float) -> None:
        active = round(shared_timeline_time, 3) != 0.0
        self.set_button.setProperty("needsOffset", active)
        self.set_button.setEnabled(active)
        self.set_button.setStyleSheet(_SET_BUTTON_ACTIVE_STYLE if active else "")
        self.timeline_changed.emit(shared_timeline_time)

    def offset(self) -> float:
        try:
            return round(float(self.spin.cleanText()), 3)
        except ValueError:
            return round(self.spin.value(), 3)


class _VideoAlignmentCard(QWidget):
    """One video viewer and its offset editor."""

    changed = Signal()
    set_requested = Signal(object)

    def __init__(self, video: Video) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.video = video
        self.load_error: OSError | None = None
        self.viewer = VideoViewer()
        self.viewer.show_overlays = False
        self.viewer.match_video_height()
        # This is here to clear viewer issues when .load goes wrong for some reason.
        try:
            self.viewer.load(video)
        except OSError as exc:
            self.load_error = exc
            self.viewer.clear()

        self.controls = _VideoAlignmentControls(video, self.viewer)
        self.shared_timeline_label = QLabel("Shared Timeline point 0.000 s")
        self.shared_timeline_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.shared_timeline_label.setStyleSheet(_TIMELINE_LABEL_STYLE)
        self.controls.timeline_changed.connect(self._show_shared_timeline_time)
        # Reset to avoid out of sync errors (when user changes tab before confirming offset or adds new input)
        if self.load_error is None:
            self.viewer.set_time_seconds(
                -video.time_offset, allow_negative=True, show_requested_time=True
            )
            self.controls._show_timeline_state(0.0)
        if self.load_error is not None:
            self.setEnabled(False)
        self.controls.spin.valueChanged.connect(lambda _value: self.changed.emit())
        self.controls.set_button.clicked.connect(
            lambda _checked=False: self.set_requested.emit(self)
        )

        layout = QVBoxLayout(self)
        self.input_label = QLabel(video.id)
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.input_label)
        layout.addWidget(self.viewer)
        layout.addWidget(self.shared_timeline_label)
        layout.addWidget(self.controls)

    @property
    def loaded(self) -> bool:
        return self.load_error is None

    def set_offset(self, offset: float) -> None:
        self.controls.set_offset(offset, preserve_timeline=False)

    def _show_shared_timeline_time(self, seconds: float) -> None:
        if seconds < 0.0:
            self.shared_timeline_label.setText(
                f"Before shared start time - will not be analyzed ({seconds:.3f} s)"
            )
            self.shared_timeline_label.setStyleSheet(_PRE_SHARED_LABEL_STYLE)
            return
        self.shared_timeline_label.setText(f"Shared Timeline point {seconds:.3f} s")
        self.shared_timeline_label.setStyleSheet(_TIMELINE_LABEL_STYLE)

    def shutdown(self) -> None:
        self.viewer.clear()


class AlignmentTab(BaseTab):
    """Let the user manually align all kind of inputs in time with each other via setting their time_offset properties."""

    title = "Alignment"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.video_cards: list[_VideoAlignmentCard] = []
        self._play_all_primary: _VideoAlignmentCard | None = None
        self.reset_timeline_button = QToolButton()
        self.reset_timeline_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self.reset_timeline_button.setToolTip("Go to timeline zero")
        self.reset_timeline_button.clicked.connect(self._go_to_timeline_zero)
        self.play_all_button = QToolButton()
        self.play_all_button.setCheckable(True)
        self.play_all_button.setIconSize(QSize(24, 24))
        self.play_all_button.setText("All")
        self.play_all_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.play_all_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_all_button.setToolTip("Play all videos")
        self.play_all_button.toggled.connect(self._on_play_all_toggled)
        self.done_button = QPushButton("Finish alignment")
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(self._finish_alignment)

        layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.video_grid_widget = QWidget()
        self.grid = QGridLayout(self.video_grid_widget)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.video_grid_widget)
        layout.addWidget(self.scroll_area, stretch=1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.reset_timeline_button)
        buttons.addWidget(self.play_all_button)
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        """Render every video input, with at most three videos per row."""
        self._stop_play_all()
        for card in self.video_cards:
            self.grid.removeWidget(card)
            card.shutdown()
            card.deleteLater()
        self.video_cards = []

        videos = [*self.experiment.glasses_videos, *self.experiment.fixed_videos]
        column_count = min(_VIDEOS_PER_ROW, max(1, len(videos)))
        for column in range(_VIDEOS_PER_ROW):
            self.grid.setColumnStretch(column, int(column < column_count))
        for index, video in enumerate(videos):
            card = _VideoAlignmentCard(video)
            if card.load_error is not None:
                self.status_message.emit(f"Could not open video: {card.load_error}")
            card.changed.connect(self.experiment_changed)
            card.set_requested.connect(self._set_offset_from_current_frame)

            self.video_cards.append(card)
            self.grid.addWidget(card, index // _VIDEOS_PER_ROW, index % _VIDEOS_PER_ROW)
        self.play_all_button.setEnabled(
            # Only enable Play-all when every card can be played to avoid possible half-playing or mid-play loading states
            bool(self.video_cards) and all(card.loaded for card in self.video_cards)
        )
        self.reset_timeline_button.setEnabled(
            any(card.loaded for card in self.video_cards)
        )

    def _set_offset_from_current_frame(self, source: _VideoAlignmentCard) -> None:
        offset = round(-source.viewer.current_time_seconds, 3)
        loaded_cards = [card for card in self.video_cards if card.loaded]
        message = QMessageBox(self)
        message.setWindowTitle("Zero current frame")
        message.setText(f"Apply offset {offset:.3f} s to this video, or to all videos?")
        message.setIcon(QMessageBox.Icon.Question)
        this_video_button = message.addButton(
            "This video", QMessageBox.ButtonRole.AcceptRole
        )
        all_videos_button = message.addButton(
            "All videos", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_button = message.addButton(QMessageBox.StandardButton.Cancel)
        this_video_button.setStyleSheet(_THIS_VIDEO_BUTTON_STYLE)
        all_videos_button.setStyleSheet(_ALL_VIDEOS_BUTTON_STYLE)
        message.setDefaultButton(this_video_button)
        message.setEscapeButton(cancel_button)
        message.exec()
        clicked_button = message.clickedButton()
        if clicked_button not in (this_video_button, all_videos_button):
            return
        self._stop_play_all()
        if clicked_button is this_video_button:
            source.set_offset(offset)
            self._show_shared_timeline_time(0.0)
            return
        for card in loaded_cards:
            card.set_offset(offset)

    def _go_to_timeline_zero(self) -> None:
        self._stop_play_all()
        self._show_shared_timeline_time(0.0)

    def _show_shared_timeline_time(self, seconds: float) -> None:
        for card in self.video_cards:
            if card.loaded:
                card.viewer.set_time_seconds(
                    seconds - card.video.time_offset,
                    allow_negative=True,
                    show_requested_time=True,
                )
                card.controls._show_timeline_state(seconds)

    def _finish_alignment(self) -> None:
        self._stop_play_all()
        self.finished.emit()

    def _on_play_all_toggled(self, play: bool) -> None:
        """
        Start or stop shared playback; all videos must be loaded.
        """
        user_desires_pause = not play
        if user_desires_pause:
            self._stop_play_all()  # Cleanly stop all
            return
        if not self.video_cards or not all(card.loaded for card in self.video_cards):
            # To prevent unusual situations with some playing
            return
        primary = self.video_cards[0]
        # This is separate because we only stop videos other than the "primary one"
        for card in self.video_cards[1:]:
            card.viewer.stop()
        self._play_all_primary = primary
        primary.viewer.frame_changed.connect(self._sync_play_all_viewers)
        self.play_all_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        )
        self.play_all_button.setToolTip("Pause all videos")
        self._sync_play_all_viewers()
        if self._play_all_primary is not None:
            primary.viewer._play_button.setChecked(True)

    def _sync_play_all_viewers(self, _frame: int = 0) -> None:
        """
        Use the primary videos current time seconds to icnrement the frames of the other videos.
        """
        primary = self._play_all_primary
        if primary is None:
            return
        shared_timeline_time = (
            primary.viewer.playback_time_seconds + primary.video.time_offset
        )
        for card in self.video_cards:
            if card is not primary:
                card.viewer.set_time_seconds(
                    shared_timeline_time - card.video.time_offset,
                    allow_negative=True,
                    show_requested_time=True,
                    sync_audio=False,
                )
                card.controls._show_timeline_state(shared_timeline_time)
        if primary.viewer.current_frame + 1 >= primary.viewer.frame_count:
            self._stop_play_all()

    def _stop_play_all(self) -> None:
        if self._play_all_primary is not None:
            self._play_all_primary.viewer.frame_changed.disconnect(
                self._sync_play_all_viewers
            )
            self._play_all_primary = None
        for card in self.video_cards:
            card.viewer.stop()
        self.play_all_button.blockSignals(True)
        self.play_all_button.setChecked(False)
        self.play_all_button.blockSignals(False)
        self.play_all_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_all_button.setToolTip("Play all videos")
