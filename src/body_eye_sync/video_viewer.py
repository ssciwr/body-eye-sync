"""A Qt widget that plays a video with frame-accurate seeking."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ultralytics.utils.plotting import colors

from body_eye_sync.state import BoundingBox


def _color_for_id(track_id: int) -> QColor:
    """Pick a colour for a track id, matching BoxMOT's own palette."""
    r, g, b = colors(int(track_id))  # ultralytics palette (RGB)
    return QColor(r, g, b)


class VideoViewer(QWidget):
    """Display a video with play/pause, a seek slider and a frame spinbox."""

    #: Emitted whenever the displayed frame changes (0-based index).
    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 0
        self._fps = 25.0
        self._current = 0
        # Index the capture will return on the next read; lets us avoid an
        # expensive seek when playing back sequentially.
        self._next_read = 0

        # Overlay drawing: a provider maps a frame index -> boxes to draw, and
        # the viewer redraws them itself whenever the frame changes.
        self._box_provider: Callable[[int], list[BoundingBox]] | None = None
        self._overlay_items: list[QGraphicsItem] = []

        # --- video display ---
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform)
        self._view.setAlignment(Qt.AlignCenter)

        # --- controls ---
        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self._on_play_toggled)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self.set_frame)

        self._spinbox = QSpinBox()
        self._spinbox.setEnabled(False)
        self._spinbox.valueChanged.connect(self.set_frame)

        self._total_label = QLabel("/ 0")

        controls = QHBoxLayout()
        controls.addWidget(self._play_button)
        controls.addWidget(self._slider, stretch=1)
        controls.addWidget(self._spinbox)
        controls.addWidget(self._total_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view, stretch=1)
        layout.addLayout(controls)

        # --- playback timer ---
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, video_path: str | Path) -> None:
        """Open ``video_path`` and display its first frame."""
        self._stop()
        if self._cap is not None:
            self._cap.release()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        self._cap = cap
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._next_read = 0
        self._timer.setInterval(int(1000 / self._fps))

        last = max(0, self._frame_count - 1)
        for control in (self._slider, self._spinbox):
            control.setEnabled(self._frame_count > 0)
            control.setMinimum(0)
            control.setMaximum(last)
        self._total_label.setText(f"/ {self._frame_count}")

        self._current = -1  # force a real read
        self.set_frame(0)
        self._fit()

    def set_frame(self, index: int) -> None:
        """Display the frame at ``index`` (0-based)."""
        if self._cap is None or self._frame_count == 0:
            return
        index = max(0, min(int(index), self._frame_count - 1))
        if index == self._current:
            return

        frame = self._read(index)
        if frame is None:
            return
        self._current = index
        self._show(frame)
        self.refresh_overlays()

        # Keep slider/spinbox in sync without re-triggering set_frame.
        for control in (self._slider, self._spinbox):
            control.blockSignals(True)
            control.setValue(index)
            control.blockSignals(False)

        self.frame_changed.emit(index)

    def set_box_provider(
        self, provider: Callable[[int], list[BoundingBox]] | None
    ) -> None:
        """Set the callable that supplies boxes for a given frame index."""
        self._box_provider = provider
        self.refresh_overlays()

    def refresh_overlays(self) -> None:
        """Redraw overlays for the current frame from the box provider."""
        self._clear_overlays()
        if self._box_provider is None or self._current < 0:
            return
        for box in self._box_provider(self._current):
            self._add_box(box)

    @property
    def current_frame(self) -> int:
        return self._current

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _read(self, index: int):
        if index != self._next_read:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._next_read = index + 1
        return frame

    def _show(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
        ).copy()  # copy so the QImage owns the buffer
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(0, 0, width, height)

    def _clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def _add_box(self, box: BoundingBox) -> None:
        color = _color_for_id(box.track_id)

        rect = QGraphicsRectItem(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
        pen = QPen(color)
        pen.setCosmetic(True)  # constant on-screen width regardless of zoom
        pen.setWidth(2)
        rect.setPen(pen)
        self._scene.addItem(rect)
        self._overlay_items.append(rect)

        label = QGraphicsSimpleTextItem(str(box.track_id))
        label.setBrush(QBrush(color))
        # Keep the label a constant size regardless of zoom/scaling.
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        label.setPos(box.x1, box.y1)
        self._scene.addItem(label)
        self._overlay_items.append(label)

    def _advance(self) -> None:
        if self._current + 1 >= self._frame_count:
            self._play_button.setChecked(False)
            return
        self.set_frame(self._current + 1)

    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText("Pause" if playing else "Play")
        if playing and self._cap is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def _stop(self) -> None:
        self._timer.stop()
        self._play_button.setChecked(False)

    def _fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit()
