"""Choose the layout of a combined video, and what goes in each of its slots."""

from __future__ import annotations

from qtpy.QtCore import QRect, Qt, Signal, Slot
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.export.layout import (
    LayoutFrame,
    LayoutKind,
    Placement,
    layout_frame,
    slot_count,
)

_EMPTY_SLOT = "(empty)"
_MINIMUM_CANVAS_HEIGHT = 220
_FRAME_COLOR = QColor(20, 20, 20)
_SLOT_ALPHA = 80
_SLOT_COLORS = (
    "#4ea3ff",
    "#00d1b2",
    "#ffb020",
    "#ff5c8a",
    "#a06bff",
    "#8bd450",
    "#ff6b4a",
    "#22d3ee",
    "#e879f9",
    "#00e676",
)


class _SlotBox(QFrame):
    """One slot of the layout: a box holding the video that fills it."""

    changed = Signal(int)

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        color = QColor(_SLOT_COLORS[index % len(_SLOT_COLORS)])
        self.setObjectName("videoLayoutSlot")
        self.setStyleSheet(
            "QFrame#videoLayoutSlot {"
            f" background-color: rgba({color.red()}, {color.green()},"
            f" {color.blue()}, {_SLOT_ALPHA});"
            f" border: 2px solid {color.name()}; }}"
        )
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo.setMinimumContentsLength(6)
        self.combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.combo.currentIndexChanged.connect(lambda _index: self.changed.emit(index))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.combo, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_videos(self, video_ids: list[str]) -> None:
        self.combo.clear()
        self.combo.addItem(_EMPTY_SLOT, None)
        for video_id in video_ids:
            self.combo.addItem(video_id, video_id)

    def video_id(self) -> str | None:
        return self.combo.currentData()

    def set_video_id(self, video_id: str | None) -> None:
        index = self.combo.findData(video_id)
        self.combo.setCurrentIndex(max(0, index))


class _LayoutCanvas(QWidget):
    """The output frame, drawn to scale, with a box placed in each slot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(_MINIMUM_CANVAS_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame = layout_frame(LayoutKind.grid, 1)
        self._boxes: list[_SlotBox] = []

    def set_slots(self, frame: LayoutFrame, boxes: list[_SlotBox]) -> None:
        self._frame = frame
        self._boxes = boxes
        self._place_boxes()
        self.update()

    def frame_rect(self) -> QRect:
        """Where the output frame sits in the widget, keeping its shape."""
        scale = min(
            self.width() / self._frame.width, self.height() / self._frame.height
        )
        width = round(scale * self._frame.width)
        height = round(scale * self._frame.height)
        return QRect(
            (self.width() - width) // 2, (self.height() - height) // 2, width, height
        )

    def _box_rect(self, placement: Placement) -> QRect:
        frame = self.frame_rect()
        scale_x = frame.width() / self._frame.width
        scale_y = frame.height() / self._frame.height
        return QRect(
            frame.x() + round(scale_x * placement.x),
            frame.y() + round(scale_y * placement.y),
            max(1, round(scale_x * placement.width)),
            max(1, round(scale_y * placement.height)),
        )

    def _place_boxes(self) -> None:
        for box, placement in zip(self._boxes, self._frame.placements):
            box.setGeometry(self._box_rect(placement))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_boxes()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.frame_rect(), _FRAME_COLOR)


class VideoLayoutEditor(QWidget):
    """Pick a layout for the combined video, and place videos in its slots."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._videos: list[str] = []
        self._assignment: list[str | None] = []
        self.boxes: list[_SlotBox] = []
        self._updating = False

        self.layout_combo = QComboBox()
        for kind in LayoutKind:
            self.layout_combo.addItem(kind.value, kind.value)
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Layout:"))
        chooser.addWidget(self.layout_combo)
        chooser.addStretch(1)

        self.canvas = _LayoutCanvas()
        self.unplaced_label = QLabel()
        self.unplaced_label.setWordWrap(True)
        self.unplaced_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(chooser)
        layout.addWidget(self.canvas)
        layout.addWidget(self.unplaced_label)
        self._rebuild()

    def layout_kind(self) -> LayoutKind:
        """The layout the combined video should use."""
        return LayoutKind(self.layout_combo.currentData())

    def set_layout_kind(self, kind: LayoutKind | str) -> None:
        index = self.layout_combo.findData(LayoutKind(kind).value)
        if index >= 0:
            self.layout_combo.setCurrentIndex(index)

    def slots(self) -> list[str | None]:
        """The video filling each slot of the layout, in drawing order."""
        return list(self._assignment)

    def set_videos(self, video_ids: list[str]) -> None:
        """Offer these videos, keeping the placements that still apply."""
        if video_ids == self._videos:
            return
        self._videos = list(video_ids)
        self._rebuild()

    def unplaced_videos(self) -> list[str]:
        """The videos the layout has no room for."""
        return [
            video_id for video_id in self._videos if video_id not in self._assignment
        ]

    def _rebuild(self) -> None:
        """Re-make the slots of the current layout, and re-fill them."""
        count = slot_count(self.layout_kind(), len(self._videos))
        assignment = [
            video_id if video_id in self._videos else None
            for video_id in self._assignment
        ][:count]
        assignment += [None] * (count - len(assignment))
        unplaced = [video_id for video_id in self._videos if video_id not in assignment]
        for index, video_id in enumerate(assignment):
            if video_id is None and unplaced:
                assignment[index] = unplaced.pop(0)
        self._assignment = assignment

        for box in self.boxes:
            box.setParent(None)
            box.deleteLater()
        self.boxes = [_SlotBox(index, self.canvas) for index in range(count)]
        for box in self.boxes:
            box.set_videos(self._videos)
            box.changed.connect(self._on_slot_changed)
            box.show()
        self._apply_assignment()
        self.canvas.set_slots(layout_frame(self.layout_kind(), count), self.boxes)

    def _apply_assignment(self) -> None:
        self._updating = True
        try:
            for box, video_id in zip(self.boxes, self._assignment):
                box.set_video_id(video_id)
        finally:
            self._updating = False
        unplaced = self.unplaced_videos()
        self.unplaced_label.setText(
            f"Not shown by this layout: {', '.join(unplaced)}" if unplaced else ""
        )
        self.unplaced_label.setVisible(bool(unplaced))

    @Slot(int)
    def _on_slot_changed(self, index: int) -> None:
        if self._updating:
            return
        video_id = self.boxes[index].video_id()
        if video_id is not None and video_id in self._assignment:
            self._assignment[self._assignment.index(video_id)] = self._assignment[index]
        self._assignment[index] = video_id
        self._apply_assignment()
        self.changed.emit()

    @Slot(int)
    def _on_layout_changed(self, _index: int) -> None:
        self._rebuild()
        self.changed.emit()
