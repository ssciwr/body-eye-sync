"""Controls shared by local-clock and synchronized media playback."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget


class PlaybackControls(QWidget):
    """A play button, position slider, and time label."""

    play_toggled = Signal(bool)
    position_requested = Signal(int)

    def __init__(
        self,
        *,
        extra_widget: QWidget | None = None,
        time_label_width: int = 105,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.toggled.connect(self._on_play_toggled)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.position_requested.emit)

        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setMinimumWidth(time_label_width)
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.play_button)
        if extra_widget is not None:
            layout.addWidget(extra_widget)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.time_label)

    def _on_play_toggled(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        self.play_toggled.emit(playing)

    def set_playing(self, playing: bool) -> None:
        """Update the button without requesting a playback change."""
        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.blockSignals(False)

    def set_range(self, start: int, end: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(start, end)
        self.slider.blockSignals(False)

    def set_position(self, position: int) -> None:
        """Move the slider without requesting a seek."""
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)

    def request_position(self, position: int) -> None:
        """Move the slider as a user action, requesting a seek."""
        self.slider.setValue(position)

    def is_seeking(self) -> bool:
        return self.slider.isSliderDown()

    def set_play_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)

    def set_seek_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(enabled)

    def set_time_text(self, text: str) -> None:
        self.time_label.setText(text)


__all__ = ["PlaybackControls"]
