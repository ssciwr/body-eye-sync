"""Non-GUI application state.

Owns the data the GUI displays (the loaded video and the tracking output) and
knows how to turn that data into something drawable for a given frame. Keeping
this Qt-free makes it straightforward to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BoundingBox:
    """A single box to draw on a frame, in video-pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    conf: float


class AppState:
    """Holds the current video and tracking results."""

    def __init__(self) -> None:
        self.video_path: Path | None = None
        self.tracklets: pd.DataFrame | None = None

    def set_video(self, path: str | Path) -> None:
        """Set the current video, invalidating any previous tracking result."""
        self.video_path = Path(path)
        self.tracklets = None

    def set_tracklets(self, tracklets: pd.DataFrame) -> None:
        self.tracklets = tracklets

    def boxes_for_frame(self, frame_index: int) -> list[BoundingBox]:
        """Return the boxes to draw on frame ``frame_index`` (0-based).

        The video viewer counts frames from 0 while BoxMOT numbers them from 1,
        so this is where the two conventions are reconciled.
        """
        if self.tracklets is None:
            return []
        rows = self.tracklets[self.tracklets["frame"] == frame_index + 1]
        return [
            BoundingBox(r.x1, r.y1, r.x2, r.y2, int(r.track_id), float(r.conf))
            for r in rows.itertuples(index=False)
        ]
