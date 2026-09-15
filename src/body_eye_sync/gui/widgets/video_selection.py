"""Select and display one video, preserving its position across refreshes."""

from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.gui.widgets.video_viewer import VideoViewer


class VideoSelectionWidget(QWidget):
    """A video chooser and viewer shared by processing tabs."""

    video_selected = Signal(object)
    status_message = Signal(str)

    def __init__(self, *, show_kind: bool = False) -> None:
        super().__init__()
        self._show_kind = show_kind
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._show_selected_video)
        self.viewer = VideoViewer()
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Video:"))
        bar.addWidget(self.selector, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.viewer, stretch=1)

    def video(self) -> Video | None:
        return self.selector.currentData()

    def set_videos(self, videos: Sequence[Video]) -> None:
        shown = self.video()
        self.selector.blockSignals(True)
        self.selector.clear()
        for video in videos:
            label = video.id
            if self._show_kind:
                kind = "glasses" if isinstance(video, GlassesVideo) else "fixed"
                label += f" ({kind})"
            self.selector.addItem(label, video)
        index = next(
            (i for i, video in enumerate(videos) if video is shown),
            0 if videos else -1,
        )
        self.selector.setCurrentIndex(index)
        self.selector.blockSignals(False)
        self.selector.setEnabled(bool(videos))
        self._show_selected_video()

    def _show_selected_video(self) -> None:
        video = self.video()
        # Tabs can configure overlays for the selection before its first frame.
        self.video_selected.emit(video)
        if video is not self.viewer.video:
            if video is None:
                self.viewer.clear()
            else:
                try:
                    self.viewer.load(video)
                except OSError as exc:
                    self.viewer.clear()
                    self.status_message.emit(f"Could not open video: {exc}")
        self.viewer.refresh_overlays()
