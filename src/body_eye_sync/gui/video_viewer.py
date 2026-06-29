"""A Qt widget that plays a video with frame-accurate seeking, and displays boxes"""

from __future__ import annotations

import cv2
from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtGui import QBrush, QImage, QPainter, QPen, QPixmap
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

from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.detection import BoundingBox, boxes_from_tracks
from body_eye_sync.gui.utils import get_color


class VideoViewer(QWidget):
    """Display a video with play/pause, a seek slider and a frame spinbox."""

    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._capture: cv2.VideoCapture | None = None
        self._frame_count = 0
        self._fps = 25.0
        self._current = 0

        # the video being displayed; supplies the boxes to draw per frame
        self._video: Video | None = None
        self._overlay_items: list[QGraphicsItem] = []

        # video display
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # controls
        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self._on_play_toggled)

        self._slider = QSlider(Qt.Orientation.Horizontal)
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

        # playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def load(self, video: Video) -> None:
        """Display ``video``, showing its first frame and its boxes (if any)."""
        self._stop()
        if self._capture is not None:
            self._capture.release()

        capture = cv2.VideoCapture(str(video.video_path))
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video.video_path}")

        self._video = video
        self._capture = capture
        self._fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        self._timer.setInterval(int(1000 / self._fps))

        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        for control in (self._slider, self._spinbox):
            control.setEnabled(count > 0)
            control.setMinimum(0)
        self._set_frame_count(count)

        self._current = -1
        self.set_frame(0)
        self._fit()

    def set_frame(self, index: int) -> None:
        """Display the frame at ``index`` (0-based), with its tracklet boxes."""
        if self._goto(index):
            self.refresh_overlays()

    @Slot(object)
    def show_live_frame(self, frame) -> None:
        """Display a freshly tracked frame and draw its boxes directly.

        Connected to the tracking worker's per-frame signal; ``frame`` is a
        BoxMOT per-frame result with 1-based indexing.
        """
        self._goto(frame.frame_idx - 1)
        self._draw_boxes(boxes_from_tracks(frame.tracks))

    def enable_controls(self, enable: bool) -> None:
        """Enable or disable the play button and seek controls."""
        if not enable:
            self._stop()
        has_frames = self._frame_count > 0
        self._play_button.setEnabled(enable and has_frames)
        self._slider.setEnabled(enable and has_frames)
        self._spinbox.setEnabled(enable and has_frames)

    def refresh_overlays(self) -> None:
        """Redraw overlays for the current frame from the video's tracklets."""
        boxes = []
        if self._video is not None and self._current >= 0:
            boxes = self._video.boxes_for_frame(self._current)
        self._draw_boxes(boxes)

    @property
    def current_frame(self) -> int:
        return self._current

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _goto(self, index: int) -> bool:
        """Show the video image at ``index`` and sync controls.

        Returns ``True`` if the displayed frame actually changed, so callers can
        decide whether overlays need redrawing.
        """
        if self._capture is None or self._frame_count == 0:
            return False
        index = max(0, min(int(index), self._frame_count - 1))
        if index == self._current:
            return False

        index, frame = self._read(index)
        if frame is None or index == self._current:
            # Nothing decoded, or _read stepped back to the frame already shown.
            return False
        self._current = index
        self._show(frame)

        # Keep slider/spinbox in sync without re-triggering set_frame.
        for control in (self._slider, self._spinbox):
            control.blockSignals(True)
            control.setValue(index)
            control.blockSignals(False)

        self.frame_changed.emit(index)
        return True

    def _draw_boxes(self, boxes: list[BoundingBox]) -> None:
        self._clear_overlays()
        for box in boxes:
            self._add_box(box)

    def _set_frame_count(self, count: int) -> None:
        """Set the frame count and update the slider/spinbox range and label."""
        self._frame_count = max(0, count)
        last = max(0, self._frame_count - 1)
        for control in (self._slider, self._spinbox):
            control.blockSignals(True)
            control.setMaximum(last)
            control.blockSignals(False)
        self._total_label.setText(f"/ {self._frame_count}")

    def _read(self, index: int):
        """Read the frame at ``index``, stepping back to the last decodable one.

        ``CAP_PROP_FRAME_COUNT`` over-estimates for many codecs, so the trailing
        frames it promises may not actually decode. When a read fails we treat
        everything from ``index`` on as non-existent, shrink the frame count to
        match, and retry the frame before it. Returns
        ``(actual_index, frame)``, or ``(-1, None)`` if nothing decodes.
        """
        # The capture cursor sits at _current + 1 after the last read, so only
        # seek (expensive) when the requested frame isn't the next one.
        sequential = index == self._current + 1
        while index >= 0:
            if not sequential:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._capture.read()
            if ok:
                return index, frame
            self._set_frame_count(index)
            index -= 1
            sequential = False
        return -1, None

    def _show(self, frame) -> None:
        height, width = frame.shape[:2]
        image = QImage(
            frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888
        )
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(0, 0, width, height)

    def _clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def _add_box(self, box: BoundingBox) -> None:
        color = get_color(box.object_id)

        rect = QGraphicsRectItem(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
        pen = QPen(color)
        # constant on-screen pen width regardless of zoom
        pen.setCosmetic(True)
        pen.setWidth(2)
        rect.setPen(pen)
        self._scene.addItem(rect)
        self._overlay_items.append(rect)

        label = QGraphicsSimpleTextItem(str(box.object_id))
        label.setBrush(QBrush(color))
        # constant on-screen label size regardless of zoom
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
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
        if playing and self._capture is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def _stop(self) -> None:
        self._timer.stop()
        self._play_button.setChecked(False)

    def _fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit()
