"""Alignment tab: placing each input on the shared experiment timeline."""

from __future__ import annotations

from qtpy.QtWidgets import QGridLayout

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets import VideoViewer


class AlignmentTab(BaseTab):
    """Let the user manually align all kind of inputs in time with each other via setting their time_offset properties."""

    title = "Alignment"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.video_viewers: list[VideoViewer] = []
        self.grid = QGridLayout(self)  # to lay out into rows of max 3
        self.refresh()  # Seems to be the paradigm to name the render function this way

    def refresh(self) -> None:
        """Render every video input, with at most three viewers per row."""
        for viewer in self.video_viewers:
            self.grid.removeWidget(viewer)
            viewer.deleteLater()
        self.video_viewers = []
        inputs_of_interest = [
            *self.experiment.glasses_videos,
            *self.experiment.fixed_videos,
        ]
        # later to include audio and paired to render below as explained

        for index, video in enumerate(
            inputs_of_interest  # later this may just remain videos and we conditionally include audio
        ):
            viewer = VideoViewer()
            viewer.show_overlays = False
            # here we will search up relevant audio if not fixed video
            try:
                viewer.load(video)
            except OSError as exc:
                self.status_message.emit(f"Could not open video: {exc}")
            self.video_viewers.append(viewer)
            self.grid.addWidget(viewer, index // 3, index % 3)

    # todo: Will be backend, but we can try using the loudest point in each audio as the possible clipboard
    # We could make it more complicated/reliable with more proper detection but I really doubt it's worth it to overcomplicate.
    # Detecting the clapper point is very much "nice to have"
