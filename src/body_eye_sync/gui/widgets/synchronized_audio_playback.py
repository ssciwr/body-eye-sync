"""Experiment-clock playback of several synchronized audio recordings."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QPointF, Qt, QTimer, QUrl, Signal, Slot
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtMultimedia import QAudioOutput, QMediaPlayer
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.utils import get_color
from body_eye_sync.gui.widgets.audio_playback import (
    _loudness_envelope,
    loudness_overview,
)
from body_eye_sync.gui.widgets.playback_controls import PlaybackControls
from body_eye_sync.media import media_duration

_RECORDING_COLOR_IDS = (0, 6, 3, 9, 18, 10, 5, 7, 13, 15, 1, 8, 17, 19)


def _time_text(seconds: float) -> str:
    sign = "−" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{int(seconds) // 60}:{seconds % 60:04.1f}"


def active_speakers(turns: pd.DataFrame | None, at: float) -> set[str]:
    """Recording IDs whose accepted speech turns contain ``at``."""
    if turns is None or turns.empty:
        return set()
    active = turns[(turns["start"] <= at) & (turns["end"] > at)]
    return set(active["speaker"].astype(str))


class _AlignedLoudnessGraph(QWidget):
    """A waveform on one shared experiment-time horizontal axis."""

    def __init__(self, start: float, end: float, color: QColor, parent=None) -> None:
        super().__init__(parent)
        self._start = start
        self._end = end
        self._values = np.empty(0, dtype=np.float32)
        self._times = np.empty(0, dtype=float)
        self._position = start
        self._color = QColor(color)
        self._message = ""
        self.setMinimumHeight(44)

    def set_values(self, values: np.ndarray, times: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float32)
        self._times = np.asarray(times, dtype=float)
        self._message = "" if self._values.size else "Loudness unavailable"
        self.update()

    def set_position(self, seconds: float) -> None:
        self._position = seconds
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        width, height = self.width(), self.height()
        middle = height / 2
        painter.setPen(QPen(self.palette().mid().color()))
        painter.drawLine(0, round(middle), width, round(middle))
        span = self._end - self._start

        if self._values.size and span > 0 and width > 0:
            xs = (self._times - self._start) / span * max(0, width - 1)
            visible = (xs >= 0) & (xs < width)
            pen = QPen(self._color)
            pen.setWidth(1)
            painter.setPen(pen)
            amplitude = self._values * max(1.0, middle - 3)
            for x, value in zip(xs[visible], amplitude[visible]):
                painter.drawLine(
                    QPointF(float(x), middle - float(value)),
                    QPointF(float(x), middle + float(value)),
                )
        elif self._message:
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)

        if span > 0:
            cursor_x = round((self._position - self._start) / span * max(0, width - 1))
            cursor_pen = QPen(self.palette().text().color())
            cursor_pen.setWidth(2)
            painter.setPen(cursor_pen)
            painter.drawLine(cursor_x, 0, cursor_x, height)


class _TrackRow(QFrame):
    """One labelled waveform, outlined while its recording contributes audio."""

    def __init__(self, name: str, start: float, end: float, color: QColor) -> None:
        super().__init__()
        self.setObjectName("synchronizedAudioTrack")
        self._color = QColor(color)
        self._label = QLabel(name)
        self._label.setMinimumWidth(100)
        self._graph = _AlignedLoudnessGraph(start, end, self._color)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.addWidget(self._label)
        layout.addWidget(self._graph, stretch=1)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        color = self._color.name() if active else "transparent"
        self.setStyleSheet(
            f"QFrame#synchronizedAudioTrack {{ border: 2px solid {color}; }}"
        )
        font = self._label.font()
        font.setBold(active)
        self._label.setFont(font)


@dataclass
class _Track:
    data: Video | Audio
    path: Path
    duration: float
    row: _TrackRow
    color: QColor
    player: QMediaPlayer | None = None
    output: QAudioOutput | None = None


class SynchronizedAudioPlaybackWidget(QWidget):
    """Stack aligned recordings and play accepted speech turns on one clock."""

    position_changed = Signal(float)
    _waveform_ready = Signal(int, str, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks: dict[str, _Track] = {}
        self._signature: tuple = ()
        self._turns: pd.DataFrame | None = None
        self._start = 0.0
        self._end = 0.0
        self._position = 0.0
        self._playing = False
        self._anchor_position = 0.0
        self._anchor_time = 0.0
        self._sync_counter = 0
        self._waveform_generation = 0
        self._waveforms_started = False

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)

        self.mute_background_checkbox = QCheckBox("Mute background")
        self.mute_background_checkbox.setChecked(True)
        self.mute_background_checkbox.toggled.connect(self._on_mute_background_toggled)
        self._controls = PlaybackControls(
            extra_widget=self.mute_background_checkbox,
            time_label_width=115,
        )
        self._controls.play_toggled.connect(self._on_play_toggled)
        self._controls.position_requested.connect(self._seek_from_controls)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._rows_layout)
        layout.addWidget(self._controls)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance)
        self._waveform_ready.connect(self._show_waveform)

    @property
    def recording_ids(self) -> list[str]:
        return list(self._tracks)

    def color_for(self, recording_id: str) -> QColor | None:
        """Return the stable display color assigned to ``recording_id``."""
        track = self._tracks.get(recording_id)
        return None if track is None else QColor(track.color)

    def load(self, recordings: list[Video | Audio]) -> None:
        """Show ``recordings`` on their shared experiment clock."""
        signature = tuple(
            (
                data.id,
                str(data.path),
                data.timeline.offset,
                data.timeline.rate,
            )
            for data in recordings
        )
        if signature == self._signature:
            return
        self.clear()
        self._signature = signature

        measured = [
            (data, media_duration(data.path))
            for data in recordings
            if data.path is not None
        ]
        measured = [(data, duration) for data, duration in measured if duration]
        if not measured:
            return
        self._start = min(data.timeline.to_experiment_time(0.0) for data, _ in measured)
        self._end = max(
            data.timeline.to_experiment_time(float(duration))
            for data, duration in measured
        )
        self._position = self._start

        for index, (data, duration) in enumerate(measured):
            color = get_color(_RECORDING_COLOR_IDS[index % len(_RECORDING_COLOR_IDS)])
            row = _TrackRow(data.id, self._start, self._end, color)
            self._rows_layout.addWidget(row)
            self._tracks[data.id] = _Track(
                data, Path(data.path), float(duration), row, color
            )

        self._controls.set_range(round(self._start * 1000), round(self._end * 1000))
        self._controls.set_position(round(self._position * 1000))
        self._controls.set_seek_enabled(True)
        self._controls.set_play_enabled(True)
        self._waveforms_started = False
        if self.isVisible():
            self._start_waveforms()
        self._show_position()

    def set_turns(self, turns: pd.DataFrame | None) -> None:
        """Set the accepted turns identifying the contributing recordings."""
        self._turns = turns
        self._update_active_speakers()

    def clear(self) -> None:
        self.pause()
        self._waveform_generation += 1
        self._waveforms_started = False
        for track in self._tracks.values():
            if track.player is not None:
                track.player.stop()
                track.player.setSource(QUrl())
                track.player.deleteLater()
            if track.output is not None:
                track.output.deleteLater()
            track.row.deleteLater()
        self._tracks.clear()
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        self._signature = ()
        self._controls.set_range(0, 0)
        self._controls.set_seek_enabled(False)
        self._controls.set_play_enabled(False)

    def play(self) -> None:
        """Start or resume all recordings from the shared position."""
        if not self._tracks:
            return
        if self._position >= self._end:
            self.seek(self._start)
        self._ensure_players()
        self._playing = True
        self._anchor_position = self._position
        self._anchor_time = time.monotonic()
        self._controls.set_playing(True)
        self._sync_players(force=True)
        self._timer.start()

    def pause(self) -> None:
        self._timer.stop()
        self._playing = False
        for track in self._tracks.values():
            if track.player is not None:
                track.player.pause()
        self._controls.set_playing(False)

    def seek(self, seconds: float) -> None:
        """Move every recording to one experiment-clock position."""
        self._position = max(self._start, min(float(seconds), self._end))
        if self._playing:
            self._anchor_position = self._position
            self._anchor_time = time.monotonic()
        self._show_position()
        self._sync_players(force=True)

    def _ensure_players(self) -> None:
        for track in self._tracks.values():
            if track.player is not None:
                continue
            output = QAudioOutput(self)
            output.setMuted(True)
            player = QMediaPlayer(self)
            player.setAudioOutput(output)
            player.setSource(QUrl.fromLocalFile(str(track.path.resolve())))
            player.setPlaybackRate(1.0 / track.data.timeline.rate)
            track.output = output
            track.player = player

    @Slot(bool)
    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            self.play()
        else:
            self.pause()

    @Slot(bool)
    def _on_mute_background_toggled(self, _checked: bool) -> None:
        self._update_active_speakers()

    @Slot(int)
    def _seek_from_controls(self, milliseconds: int) -> None:
        self.seek(milliseconds / 1000)

    def _advance(self) -> None:
        position = self._anchor_position + (time.monotonic() - self._anchor_time)
        if position >= self._end:
            self._position = self._end
            self._show_position()
            self.pause()
            return
        self._position = position
        self._show_position()
        self._sync_counter += 1
        if self._sync_counter % 10 == 0:
            self._sync_players(force=False)

    def _show_position(self) -> None:
        self._controls.set_position(round(self._position * 1000))
        self._controls.set_time_text(
            f"{_time_text(self._position)} / {_time_text(self._end)}"
        )
        for track in self._tracks.values():
            track.row._graph.set_position(self._position)
        self._update_active_speakers()
        self.position_changed.emit(self._position)

    def _update_active_speakers(self) -> None:
        active = active_speakers(self._turns, self._position)
        for name, track in self._tracks.items():
            contributes = name in active
            track.row.set_active(contributes)
            if track.output is not None:
                local = track.data.timeline.to_local_time(self._position)
                covered = local is not None and 0 <= local <= track.duration
                track.output.setMuted(
                    not covered
                    or (self.mute_background_checkbox.isChecked() and not contributes)
                )

    def _sync_players(self, *, force: bool) -> None:
        if not self._playing:
            return
        active = active_speakers(self._turns, self._position)
        for name, track in self._tracks.items():
            if track.player is None or track.output is None:
                continue
            local = track.data.timeline.to_local_time(self._position)
            covered = local is not None and 0 <= local <= track.duration
            track.output.setMuted(
                not covered
                or (self.mute_background_checkbox.isChecked() and name not in active)
            )
            if not covered:
                track.player.pause()
                continue
            target = round(local * 1000)
            if force or abs(track.player.position() - target) > 80:
                track.player.setPosition(target)
            track.player.play()

    def _start_waveforms(self) -> None:
        if self._waveforms_started:
            return
        self._waveforms_started = True
        generation = self._waveform_generation
        for name, track in self._tracks.items():
            threading.Thread(
                target=self._decode_waveform,
                args=(generation, name, track),
                daemon=True,
                name=f"loudness-{name}",
            ).start()

    def _decode_waveform(self, generation: int, name: str, track: _Track) -> None:
        try:
            levels = track.data.loudness.levels
            values = (
                loudness_overview(levels)
                if levels.size
                else _loudness_envelope(track.path)
            )
            times = track.data.timeline.to_experiment_times(
                np.linspace(0.0, track.duration, len(values), endpoint=False)
            )
        except Exception:
            values, times = np.empty(0), np.empty(0)
        try:
            self._waveform_ready.emit(generation, name, values, times)
        except RuntimeError:
            pass

    @Slot(int, str, object, object)
    def _show_waveform(self, generation: int, name: str, values, times) -> None:
        if generation != self._waveform_generation or name not in self._tracks:
            return
        self._tracks[name].row._graph.set_values(values, times)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_waveforms()


__all__ = ["SynchronizedAudioPlaybackWidget", "active_speakers"]
