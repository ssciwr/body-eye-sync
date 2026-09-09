"""Audio playback controls with a compact loudness overview."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF, Qt, QUrl, Signal, Slot
from qtpy.QtGui import QMouseEvent, QPainter, QPen, QPolygonF
from qtpy.QtMultimedia import QAudioOutput, QMediaPlayer
from qtpy.QtWidgets import QVBoxLayout, QWidget

from body_eye_sync.gui.widgets.playback_controls import PlaybackControls


def _time_text(milliseconds: int) -> str:
    """Format a media position as ``m:ss.s``."""
    seconds = max(0, milliseconds) / 1000
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


def _normalised(rms: np.ndarray) -> np.ndarray:
    """Scale RMS levels into 0..1 across the top 60 dB of the recording."""
    if not np.any(rms > np.finfo(np.float32).eps):
        return np.zeros(rms.size, dtype=np.float32)
    decibels = 20 * np.log10(np.maximum(rms, np.finfo(np.float32).tiny))
    ceiling = float(np.max(decibels))
    return np.clip((decibels - (ceiling - 60.0)) / 60.0, 0.0, 1.0)


def loudness_overview(levels: np.ndarray, points: int = 1200) -> np.ndarray:
    """Return a normalized loudness overview to plot from levels measured in dB."""
    levels = np.asarray(levels, dtype=float)
    if levels.size == 0:
        return np.empty(0, dtype=np.float32)
    count = min(points, levels.size)
    edges = np.linspace(0, levels.size, count + 1, dtype=np.int64)
    # Each bar is the RMS of the chunks it covers, as decoding would give it.
    power = np.add.reduceat(10.0 ** (levels / 10.0), edges[:-1])
    return _normalised(np.sqrt(power / np.diff(edges)).astype(np.float32))


def _loudness_envelope(path: Path, points: int = 1200) -> np.ndarray:
    """Return a normalized RMS loudness overview for an audio-bearing file."""
    from body_eye_sync.preprocessing.audio import load_audio

    samples = np.asarray(load_audio(path), dtype=np.float32)
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)

    count = min(points, samples.size)
    edges = np.linspace(0, samples.size, count + 1, dtype=np.int64)
    rms = np.empty(count, dtype=np.float32)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        window = samples[start:end]
        rms[index] = np.sqrt(np.mean(window * window))
    return _normalised(rms)


class _LoudnessGraph(QWidget):
    """Paint a loudness envelope and playback cursor; clicking seeks."""

    seek_requested = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values = np.empty(0, dtype=np.float32)
        self._position = 0.0
        self._message = ""
        self.setMinimumHeight(64)
        self.setToolTip("Audio loudness; click or drag to seek")

    def set_values(self, values: np.ndarray, message: str = "") -> None:
        self._values = np.asarray(values, dtype=np.float32)
        self._message = message
        self.update()

    def set_position(self, fraction: float) -> None:
        fraction = max(0.0, min(float(fraction), 1.0))
        if fraction != self._position:
            self._position = fraction
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())

        width = self.width()
        height = self.height()
        middle = height / 2
        painter.setPen(QPen(self.palette().mid().color()))
        painter.drawLine(0, round(middle), width, round(middle))

        if self._values.size:
            xs = np.arange(width, dtype=np.float32)
            source_xs = np.linspace(0, max(0, width - 1), self._values.size)
            values = np.interp(xs, source_xs, self._values)
            amplitude = values * max(1.0, middle - 4)
            polygon = QPolygonF(
                [QPointF(float(x), middle - float(y)) for x, y in zip(xs, amplitude)]
                + [
                    QPointF(float(x), middle + float(y))
                    for x, y in zip(xs[::-1], amplitude[::-1])
                ]
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.palette().highlight())
            painter.drawPolygon(polygon)
        elif self._message:
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)

        cursor_x = round(self._position * max(0, width - 1))
        cursor_pen = QPen(self.palette().text().color())
        cursor_pen.setWidth(2)
        painter.setPen(cursor_pen)
        painter.drawLine(cursor_x, 0, cursor_x, height)

    def _seek(self, event: QMouseEvent) -> None:
        if self.width() > 0:
            fraction = max(0.0, min(event.position().x() / self.width(), 1.0))
            self.seek_requested.emit(fraction)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._seek(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek(event)
            event.accept()
            return
        super().mouseMoveEvent(event)


class AudioPlaybackWidget(QWidget):
    """Play an audio-bearing file, seek it, and display its loudness."""

    position_changed = Signal(float)
    _waveform_ready = Signal(int, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._levels = np.empty(0)
        self._duration = 0
        self._waveform_generation = 0
        self._waveform_started = False

        self._audio_output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._controls = PlaybackControls()
        self._controls.play_toggled.connect(self._on_play_toggled)
        self._controls.position_requested.connect(self._seek)

        self._graph = _LoudnessGraph()
        self._graph.seek_requested.connect(self._seek_fraction)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._graph)
        layout.addWidget(self._controls)

        self._waveform_ready.connect(self._show_waveform)

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self, path: str | Path, levels: np.ndarray | None = None) -> None:
        """Load an audio file or the audio track of a video file.

        ``levels`` are the recording's loudness in dB, if not provided will be measured in a background thread.
        """
        path = Path(path)
        levels = np.empty(0) if levels is None else np.asarray(levels, dtype=float)
        if path == self._path and levels.size == self._levels.size:
            return
        self.clear()
        self._path = path
        self._levels = levels
        self._waveform_generation += 1
        self._waveform_started = False
        self._graph.set_values(np.empty(0), "")
        self._controls.set_play_enabled(True)
        self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        if self.isVisible():
            self._start_waveform()

    def clear(self) -> None:
        """Stop playback and forget the currently loaded recording."""
        self.pause()
        self._path = None
        self._duration = 0
        self._waveform_generation += 1
        self._waveform_started = False
        self._player.setSource(QUrl())
        self._controls.set_range(0, 0)
        self._controls.set_seek_enabled(False)
        self._controls.set_play_enabled(False)
        self._graph.set_values(np.empty(0), "")
        self._graph.set_position(0.0)
        self._update_time(0)

    def pause(self) -> None:
        self._player.pause()
        self._controls.set_playing(False)

    def play(self) -> None:
        """Start or resume playback of the loaded recording."""
        if self._path is not None:
            self._controls.play_button.setChecked(True)

    def seek(self, seconds: float) -> None:
        """Move playback to ``seconds`` on the recording clock."""
        self._player.setPosition(round(max(0.0, seconds) * 1000))

    @Slot(bool)
    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            if self._duration and self._player.position() >= self._duration:
                self._player.setPosition(0)
            self._player.play()
        else:
            self._player.pause()

    @Slot(object)
    def _on_playback_state_changed(self, state) -> None:
        self._controls.set_playing(state == QMediaPlayer.PlaybackState.PlayingState)

    @Slot(int)
    def _on_duration_changed(self, milliseconds: int) -> None:
        self._duration = max(0, milliseconds)
        self._controls.set_range(0, self._duration)
        self._controls.set_seek_enabled(self._path is not None and self._duration > 0)
        self._update_time(self._player.position())

    @Slot(int)
    def _on_position_changed(self, milliseconds: int) -> None:
        if not self._controls.is_seeking():
            self._controls.set_position(milliseconds)
        self._update_time(milliseconds)
        self._graph.set_position(
            milliseconds / self._duration if self._duration else 0.0
        )
        self.position_changed.emit(milliseconds / 1000)

    @Slot(int)
    def _seek(self, milliseconds: int) -> None:
        self._player.setPosition(milliseconds)
        self._update_time(milliseconds)
        self._graph.set_position(
            milliseconds / self._duration if self._duration else 0.0
        )
        self.position_changed.emit(milliseconds / 1000)

    @Slot(float)
    def _seek_fraction(self, fraction: float) -> None:
        self._controls.request_position(round(fraction * self._duration))

    def _update_time(self, position: int) -> None:
        self._controls.set_time_text(
            f"{_time_text(position)} / {_time_text(self._duration)}"
        )

    def _start_waveform(self) -> None:
        if self._path is None or self._waveform_started:
            return
        self._waveform_started = True
        generation = self._waveform_generation
        if self._levels.size:
            self._show_waveform(generation, loudness_overview(self._levels))
            return
        path = self._path
        threading.Thread(
            target=self._decode_waveform,
            args=(generation, path),
            daemon=True,
            name="audio-loudness",
        ).start()

    def _decode_waveform(self, generation: int, path: Path) -> None:
        try:
            values = _loudness_envelope(path)
        except Exception:
            values = None
        try:
            self._waveform_ready.emit(generation, values)
        except RuntimeError:
            pass

    @Slot(int, object)
    def _show_waveform(self, generation: int, values) -> None:
        if generation != self._waveform_generation:
            return
        if values is None:
            self._graph.set_values(np.empty(0), "Loudness unavailable")
        elif len(values) == 0:
            self._graph.set_values(np.empty(0), "No audio samples")
        else:
            self._graph.set_values(values)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_waveform()


__all__ = ["AudioPlaybackWidget", "loudness_overview"]
