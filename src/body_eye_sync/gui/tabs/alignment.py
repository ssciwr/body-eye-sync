"""Alignment tab: placing each input on the shared experiment timeline."""

from __future__ import annotations

from qtpy.QtCore import QUrl
from qtpy.QtMultimedia import QAudioOutput, QMediaPlayer
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import GlassesVideo
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import VideoViewer


# Minimal player used in the alignment tab before waveform/timeline work exists.
class _AudioPlayer(QWidget):
    """Play one audio input. No audio waveforms yet."""

    def __init__(self, audio: Audio) -> None:
        super().__init__()
        self.audio = audio
        self.output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.setSource(QUrl.fromLocalFile(str(audio.audio_path)))

        self.button = QPushButton(f"Play {audio.id}")
        self.button.clicked.connect(self._toggle)
        layout = QVBoxLayout(self)
        layout.addWidget(self.button)

    # Toggle playback for the minimal audio player.
    def _toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.button.setText("Play")
            return
        self.player.play()
        self.button.setText("Pause")
        # Later this could offer "Play from offset" rather than resuming in place.


class AlignmentTab(BaseTab):
    """Let the user manually align all kind of inputs in time with each other via setting their time_offset properties."""

    title = "Alignment"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self._cells: list[QWidget] = []
        self.video_viewers: list[VideoViewer] = []
        self.audio_players: list[_AudioPlayer] = []
        self.no_paired_audio_labels: list[QLabel] = []
        self.unpaired_audio_labels: list[QLabel] = []
        self._show_audio = False

        self.step_label = QLabel("Step 1: Align all videos")
        self.done_button = QPushButton("Finish videos, align audio")
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(self._start_audio_step)

        layout = QVBoxLayout(self)
        layout.addWidget(self.step_label)
        self.grid = QGridLayout()  # to lay out into rows of max 3
        layout.addLayout(self.grid, stretch=1)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.done_button)
        layout.addLayout(button_row)
        self.refresh()  # Seems to be the paradigm to name the render function this way

    def refresh(self) -> None:
        """Render every video input, with at most three viewers per row."""
        for widget in self._cells:
            self.grid.removeWidget(widget)
            widget.deleteLater()
        self._cells = []
        self.video_viewers = []
        self.audio_players = []
        self.no_paired_audio_labels = []
        self.unpaired_audio_labels = []
        inputs_of_interest = [
            *self.experiment.glasses_videos,
            *self.experiment.fixed_videos,
        ]

        for index, video in enumerate(inputs_of_interest):
            cell = QWidget()
            self._cells.append(cell)
            cell_layout = QVBoxLayout(cell)
            viewer = VideoViewer()
            viewer.show_overlays = False
            try:
                viewer.load(video)
            except OSError as exc:
                self.status_message.emit(f"Could not open video: {exc}")
            self.video_viewers.append(viewer)
            cell_layout.addWidget(viewer)

            if self._show_audio and isinstance(video, GlassesVideo):
                paired_audio = [
                    audio
                    for audio in self.experiment.audio
                    if audio.glasses_video is video
                ]
                if not paired_audio:
                    label = QLabel("No paired audio")
                    label.setEnabled(False)
                    self.no_paired_audio_labels.append(label)
                    cell_layout.addWidget(label)
                for audio in paired_audio:
                    player = _AudioPlayer(audio)
                    self.audio_players.append(player)
                    cell_layout.addWidget(player)

            self.grid.addWidget(cell, index // 3, index % 3)

        if self._show_audio:
            unpaired_audio = [
                audio for audio in self.experiment.audio if audio.glasses_video is None
            ]
            for offset, audio in enumerate(unpaired_audio):
                index = len(inputs_of_interest) + offset
                cell = QWidget()
                self._cells.append(cell)
                cell_layout = QVBoxLayout(cell)
                label = QLabel("Unpaired audio")
                self.unpaired_audio_labels.append(label)
                cell_layout.addWidget(label)
                player = _AudioPlayer(audio)
                self.audio_players.append(player)
                cell_layout.addWidget(player)
                self.grid.addWidget(cell, index // 3, index % 3)

    # Move into the audio step after the user has accepted the video alignment.
    def _start_audio_step(self) -> None:
        self._show_audio = True
        self.step_label.setText("Step 2: Check the audio is aligned to its video")
        self.done_button.setVisible(False)
        self.refresh()
