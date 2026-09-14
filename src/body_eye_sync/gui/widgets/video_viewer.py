"""A Qt widget that plays a video with frame-accurate seeking, and displays boxes"""

from __future__ import annotations

from math import isfinite

import cv2
from qtpy.QtCore import Qt, QTimer, QUrl, Signal, Slot
from qtpy.QtGui import QBrush, QFont, QImage, QPainter, QPen, QPixmap
from qtpy.QtMultimedia import QAudioOutput, QMediaPlayer
from qtpy.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.object_tracking import BoundingBox, boxes_from_tracks
from body_eye_sync.pipeline.body_pose import SKELETON, BodyPose
from body_eye_sync.pipeline.face_detection import FaceBox
from body_eye_sync.gui.utils import get_color

_MINIMUM_VIDEO_VIEW_HEIGHT = 80
_PLAYBACK_POLL_INTERVAL_MS = 10
_MAX_SEQUENTIAL_FORWARD_FRAMES = 10


class _VideoGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.allow_parent_scroll = False

    def wheelEvent(self, event) -> None:
        if self.allow_parent_scroll:
            event.ignore()
            return
        super().wheelEvent(event)


class VideoViewer(QWidget):
    """Display a video with play/pause, a seek slider and a frame spinbox."""

    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._capture: cv2.VideoCapture | None = None
        self._frame_count = 0
        self._fps = 25.0
        self._current = 0
        self._preroll_seconds: float | None = None
        self._displayed_time_seconds: float | None = None
        self._video_aspect_ratio: float | None = None
        self._height_matches_video = False
        self._audio_output = QAudioOutput(self)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)

        # the video being displayed; supplies the boxes to draw per frame
        self._video: Video | None = None
        self.show_overlays = True
        self._overlay_items: list[QGraphicsItem] = []

        # video display
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._view = _VideoGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # controls
        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self._on_play_toggled)

        self._mute_button = QToolButton()
        self._mute_button.setCheckable(True)
        self._mute_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
        )
        self._mute_button.setToolTip(
            "Mute audio"
        )  # gets updated later as state changes.
        self._mute_button.toggled.connect(self._on_mute_toggled)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self.set_frame)

        self._spinbox = QSpinBox()
        self._spinbox.setEnabled(False)
        self._spinbox.valueChanged.connect(self.set_frame)

        self._time_label = QLabel("0.000 s")
        self._total_label = QLabel("/ 0")

        controls = QHBoxLayout()
        controls.addWidget(self._play_button)
        controls.addWidget(self._mute_button)
        controls.addWidget(self._slider, stretch=1)
        controls.addWidget(self._time_label)
        controls.addWidget(self._spinbox)
        controls.addWidget(self._total_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view, stretch=1)
        layout.addLayout(controls)

        # playback timer
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance)

    def match_video_height(self) -> None:
        self._height_matches_video = True
        self._view.allow_parent_scroll = True
        self._view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.match_container_height_to_video_height()

    def load(self, video: Video) -> None:
        """Display ``video``, showing its first frame and its boxes (if any)."""
        self.stop()
        if self._capture is not None:
            self._capture.release()

        capture = cv2.VideoCapture(str(video.video_path))
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video.video_path}")

        self._video = video
        self._capture = capture
        self._media_player.setSource(QUrl.fromLocalFile(str(video.video_path)))
        self._fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        self._timer.setInterval(max(1, round(1000 / self._fps)))

        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        for control in (self._slider, self._spinbox):
            control.setEnabled(count > 0)
            control.setMinimum(0)
        self._set_frame_count(count)

        self._current = -1
        self._preroll_seconds = None
        self.set_frame(0)
        self.fit_image_at_aspect_ratio()

    def clear(self) -> None:
        """Show nothing at all: no video, no frame and no overlays."""
        self.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._video = None
        self._current = -1
        self._preroll_seconds = None
        self._displayed_time_seconds = None
        self._video_aspect_ratio = None
        self._media_player.stop()
        self._media_player.setSource(QUrl())
        self._clear_overlays()
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, 0, 0)
        self._set_frame_count(0)
        self.enable_controls(False)

    def set_frame(
        self,
        index: int,
        *,
        displayed_time_seconds: float | None = None,
        sync_audio: bool = True,
    ) -> None:
        """Display frame ``index`` and optionally seek embedded audio to it."""
        previous_time_seconds = self.current_time_seconds
        self._displayed_time_seconds = displayed_time_seconds
        if self._goto(index, sync_audio=sync_audio):
            self.refresh_overlays()
            return
        current_time_seconds = self.current_time_seconds
        if (
            displayed_time_seconds is not None
            or current_time_seconds != previous_time_seconds
        ):
            self._time_label.setText(f"{current_time_seconds:.3f} s")
        if sync_audio and current_time_seconds != previous_time_seconds:
            self._sync_audio_to_frame()
        if (
            current_time_seconds != previous_time_seconds
            and self._capture is not None
            and self._frame_count > 0
        ):
            self.frame_changed.emit(self._current)

    # Display the frame closest to ``seconds`` in the video.
    def set_time_seconds(
        self,
        seconds: float,
        *,
        allow_negative: bool = False,
        show_requested_time: bool = False,
        sync_audio: bool = True,
    ) -> None:
        """Display the frame selected by ``seconds`` in the source video."""
        if self._fps <= 0.0:
            self.set_frame(0, sync_audio=sync_audio)
            return
        if allow_negative and seconds < 0.0:
            self._show_preroll_frame(seconds)
            return
        frame = (
            int(seconds * self._fps)
            if show_requested_time
            else round(seconds * self._fps)
        )
        self.set_frame(
            max(0, frame),
            displayed_time_seconds=seconds if show_requested_time else None,
            sync_audio=sync_audio,
        )

    @Slot(object)
    def show_live_frame(self, frame) -> None:
        """Display a freshly tracked frame and draw its boxes directly.

        Connected to the object tracking worker's per-frame signal; ``frame`` is
        a BoxMOT per-frame result with 1-based indexing.
        """
        self._goto(frame.frame_idx - 1)
        self._draw_boxes(boxes_from_tracks(frame.tracks))

    @Slot(object)
    def show_live_face_frame(self, result) -> None:
        """Display a freshly face-detected frame, with person boxes and faces.

        Connected to the face-detection worker's per-frame signal; ``result`` is
        a :class:`FaceFrameResult` with 0-based indexing. The person boxes come
        from the already-tracked video, the faces straight from the result.
        """
        self._goto(result.frame_idx)
        self._clear_overlays()
        if not self.show_overlays:
            return
        if self._video is not None:
            for box in self._video.boxes_for_frame(self._current):
                self._add_box(box)
        for face in result.faces:
            self._add_face(face)

    @Slot(object)
    def show_live_pose_frame(self, result) -> None:
        """Display a freshly pose-detected frame, with person boxes and poses.

        Connected to the body-pose worker's per-frame signal; ``result`` is a
        :class:`PoseFrameResult` with 0-based indexing. The person boxes come
        from the already-tracked video, the poses straight from the result.
        """
        self._goto(result.frame_idx)
        self._clear_overlays()
        if not self.show_overlays:
            return
        if self._video is not None:
            for box in self._video.boxes_for_frame(self._current):
                self._add_box(box)
        for pose in result.poses:
            self._add_pose(pose)

    def enable_controls(self, enable: bool) -> None:
        """Enable or disable the playback, mute and seek controls."""
        if not enable:
            self.stop()
        has_frames = self._frame_count > 0
        self._play_button.setEnabled(enable and has_frames)
        self._mute_button.setEnabled(enable and has_frames)
        self._slider.setEnabled(enable and has_frames)
        self._spinbox.setEnabled(enable and has_frames)

    def refresh_overlays(self) -> None:
        """Redraw the current frame's person boxes and any detected faces."""
        self._clear_overlays()
        if not self.show_overlays or self._video is None or self._current < 0:
            return
        for box in self._video.boxes_for_frame(self._current):
            self._add_box(box)
        for pose in self._video.poses_for_frame(self._current):
            self._add_pose(pose)
        for face in self._video.faces_for_frame(self._current):
            self._add_face(face)

    @property
    def video(self) -> Video | None:
        """The video being displayed, or ``None`` if there is none."""
        return self._video

    @property
    def current_frame(self) -> int:
        return self._current

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # The playback position represented by the current frame.
    @property
    def current_time_seconds(self) -> float:
        if self._preroll_seconds is not None:
            return self._preroll_seconds
        if self._displayed_time_seconds is not None:
            return self._displayed_time_seconds
        if self._frame_count == 0 or self._fps <= 0.0:
            return 0.0
        return self._current / self._fps

    @property
    def current_media_time_seconds(self) -> float:
        """Timestamp of the displayed video frame in the source media."""
        if self._frame_count == 0 or self._fps <= 0.0 or self._current < 0:
            return 0.0
        return self._current / self._fps

    @property
    def playback_time_seconds(self) -> float:
        """Exact source-media time represented by the playback clock."""
        if self._timer.isActive() and self._preroll_seconds is None:
            return self._media_position_seconds()
        return self.current_time_seconds

    def _goto(self, index: int, *, sync_audio: bool = True) -> bool:
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
        self._preroll_seconds = None
        self._current = index
        self._show(frame)

        # Keep slider/spinbox in sync without re-triggering set_frame.
        for control in (self._slider, self._spinbox):
            control.blockSignals(True)
            control.setValue(index)
            control.blockSignals(False)
        self._time_label.setText(f"{self.current_time_seconds:.3f} s")
        if sync_audio:
            self._sync_audio_to_frame()

        self.frame_changed.emit(index)
        return True

    def _draw_boxes(self, boxes: list[BoundingBox]) -> None:
        self._clear_overlays()
        if not self.show_overlays:
            return
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

        Small forward jumps decode and discard the intervening frames because
        that is substantially cheaper than seeking in compressed video. Larger
        jumps and all backward moves seek directly.

        ``CAP_PROP_FRAME_COUNT`` over-estimates for many codecs, so the trailing
        frames it promises may not actually decode. When a read fails we treat
        everything from ``index`` on as non-existent, shrink the frame count to
        match, and retry the frame before it. Returns
        ``(actual_index, frame)``, or ``(-1, None)`` if nothing decodes.
        """
        forward_frames = index - self._current
        if self._current >= 0 and 1 <= forward_frames <= _MAX_SEQUENTIAL_FORWARD_FRAMES:
            last_index = -1
            last_frame = None
            for candidate in range(self._current + 1, index + 1):
                ok, frame = self._capture.read()
                if not ok:
                    self._set_frame_count(candidate)
                    break
                last_index = candidate
                last_frame = frame
            return last_index, last_frame

        while index >= 0:
            if self._capture.get(cv2.CAP_PROP_POS_FRAMES) != index:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._capture.read()
            if ok:
                return index, frame
            self._set_frame_count(index)
            index -= 1
        return -1, None

    def _show(self, frame) -> None:
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Video frame has no size")
        self._video_aspect_ratio = width / height
        self.match_container_height_to_video_height()
        image = QImage(
            frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888
        )
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(0, 0, width, height)

    # Show the waiting period before a positively-offset video starts.
    def _show_preroll_frame(self, seconds: float) -> None:
        self._preroll_seconds = seconds
        self._current = min(-1, int(seconds * self._fps))
        self._media_player.pause()
        pixmap = QPixmap(self._pixmap_item.pixmap().size())
        pixmap.fill(Qt.GlobalColor.black)
        self._clear_overlays()
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        text = QGraphicsSimpleTextItem(f"{self.current_time_seconds:.3f} s")
        text.setFont(QFont(self.font().family(), 50))
        text.setBrush(QBrush(Qt.GlobalColor.white))
        text.setPos(
            (pixmap.width() - text.boundingRect().width()) / 2,
            (pixmap.height() - text.boundingRect().height()) / 2,
        )
        self._scene.addItem(text)
        self._overlay_items.append(text)
        self._time_label.setText(f"{self.current_time_seconds:.3f} s")
        for control in (self._slider, self._spinbox):
            control.blockSignals(True)
            control.setValue(0)
            control.blockSignals(False)
        self.frame_changed.emit(self._current)

    def _clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def _add_rect(self, box: BoundingBox, style: Qt.PenStyle) -> None:
        """Draw ``box`` as a rectangle coloured by its id, in the given pen style."""
        rect = QGraphicsRectItem(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
        pen = QPen(get_color(box.track_id))
        pen.setStyle(style)
        # constant on-screen pen width regardless of zoom
        pen.setCosmetic(True)
        pen.setWidth(2)
        rect.setPen(pen)
        self._scene.addItem(rect)
        self._overlay_items.append(rect)

    def _add_box(self, box: BoundingBox) -> None:
        self._add_rect(box, Qt.PenStyle.SolidLine)

        label = QGraphicsSimpleTextItem(str(box.track_id))
        label.setBrush(QBrush(get_color(box.track_id)))
        # constant on-screen label size regardless of zoom
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        label.setPos(box.x1, box.y1)
        self._scene.addItem(label)
        self._overlay_items.append(label)

    def _add_face(self, face: FaceBox) -> None:
        # dashed, so the face box reads as distinct from its person box
        self._add_rect(face.box, Qt.PenStyle.DashLine)

        color = get_color(face.box.track_id)
        for px, py in face.landmarks:
            # a small constant-size dot regardless of zoom, centred on the point
            dot = QGraphicsEllipseItem(-2.0, -2.0, 4.0, 4.0)
            dot.setBrush(QBrush(color))
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            dot.setPos(px, py)
            self._scene.addItem(dot)
            self._overlay_items.append(dot)

    def _add_pose(self, pose: BodyPose) -> None:
        color = get_color(pose.box.track_id)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(2)

        visible = [
            score > 0.0 and isfinite(px) and isfinite(py)
            for px, py, score in pose.keypoints
        ]
        for start, end in SKELETON:
            if start >= len(pose.keypoints) or end >= len(pose.keypoints):
                continue
            if not (visible[start] and visible[end]):
                continue
            x1, y1, _ = pose.keypoints[start]
            x2, y2, _ = pose.keypoints[end]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(pen)
            self._scene.addItem(line)
            self._overlay_items.append(line)

        for px, py, score in pose.keypoints:
            if not (score > 0.0 and isfinite(px) and isfinite(py)):
                continue
            dot = QGraphicsEllipseItem(-2.0, -2.0, 4.0, 4.0)
            dot.setBrush(QBrush(color))
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            dot.setPos(px, py)
            self._scene.addItem(dot)
            self._overlay_items.append(dot)

    def _advance(self) -> None:
        if self._preroll_seconds is not None:
            next_seconds = self._preroll_seconds + 1 / self._fps
            if next_seconds < 0.0:
                self._show_preroll_frame(next_seconds)
                return
            self.set_frame(0)
            if self._play_button.isChecked():
                self._start_media_playback()
            return
        target_frame = self._media_frame_index()
        if target_frame >= self._frame_count:
            if self._current < self._frame_count - 1:
                self.set_frame(self._frame_count - 1, sync_audio=False)
            self._play_button.setChecked(False)
            return
        if target_frame <= self._current:
            return
        self.set_frame(target_frame, sync_audio=False)

    def _media_frame_index(self) -> int:
        """Return the frame containing the media player's current position."""
        return max(0, int(self._media_position_seconds() * self._fps))

    def _media_position_seconds(self) -> float:
        return self._media_player.position() / 1000

    def _start_media_playback(self) -> None:
        self._timer.setInterval(_PLAYBACK_POLL_INTERVAL_MS)
        self._sync_audio_to_frame()
        self._media_player.play()

    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText("Pause" if playing else "Play")
        if playing and self._capture is not None:
            if self._current >= 0:
                self._start_media_playback()
            else:
                self._timer.setInterval(max(1, round(1000 / self._fps)))
            self._timer.start()
        else:
            self._timer.stop()
            self._media_player.pause()

    def _on_mute_toggled(self, muted: bool) -> None:
        self._audio_output.setMuted(muted)
        icon = (
            QStyle.StandardPixmap.SP_MediaVolumeMuted
            if muted
            else QStyle.StandardPixmap.SP_MediaVolume  # just looks empty, noticeably not activated vs the other one.
        )
        label = "Unmute audio" if muted else "Mute audio"
        self._mute_button.setIcon(self.style().standardIcon(icon))
        self._mute_button.setToolTip(label)

    def stop(self) -> None:
        self._timer.stop()
        self._media_player.pause()
        self._play_button.setChecked(False)

    # Seek to an exact requested time when present, otherwise to the frame time.
    def _sync_audio_to_frame(self) -> None:
        self._media_player.setPosition(round(self.current_time_seconds * 1000))
        # see experiments.md for notes about when this audio could be out of sync with the same files video.

    def fit_image_at_aspect_ratio(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def match_container_height_to_video_height(self) -> None:
        # Make sure the container height of the wideget matches the actual videos hegiht.
        if not self._height_matches_video or self._video_aspect_ratio is None:
            return
        border_width = self._view.frameWidth()
        view_width = self._view.width()
        if view_width <= 0:
            view_width = self.width()
        video_width = max(1, view_width - 2 * border_width)
        video_height = max(
            _MINIMUM_VIDEO_VIEW_HEIGHT, round(video_width / self._video_aspect_ratio)
        )
        view_height = video_height + 2 * border_width
        if self._view.height() != view_height:
            self._view.setFixedHeight(view_height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.match_container_height_to_video_height()
        self.fit_image_at_aspect_ratio()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.match_container_height_to_video_height()
        self.fit_image_at_aspect_ratio()
