"""Object tracking and vision model outputs for a video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from body_eye_sync.pipeline.detection import BoundingBox, tracks_to_dataframe


class Video:
    """Contains object tracking and vision model outputs for a video.

    Completed results live in a single numeric :attr:`data` DataFrame. While a
    run is in progress, each frame's BoxMOT ``tracks`` array is accumulated and
    collapsed into that DataFrame once :meth:`finish_tracking` is called.
    """

    def __init__(self) -> None:
        self.video_path: Path | None = None
        self._data: pd.DataFrame | None = None
        self._rows_by_frame: dict[int, np.ndarray] = {}
        self._frames: list[tuple[int, np.ndarray]] = []

    def set_video(self, path: str | Path) -> None:
        """Set the current video, invalidating any previous tracking result."""
        self.video_path = Path(path)
        self.clear()

    def begin_tracking(self) -> None:
        self.clear()

    def add_frame(self, frame) -> None:
        """Accumulate a BoxMOT per-frame result, converting to 0-based indices"""
        self._frames.append((frame.frame_idx - 1, np.asarray(frame.tracks)))

    def finish_tracking(self) -> None:
        """Collapse the streamed frames into the stored :attr:`data` DataFrame."""
        self.set_data(tracks_to_dataframe(self._frames))

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace all results with a complete data DataFrame."""
        self._data = data
        self._rows_by_frame = data.groupby("frame").indices
        self._frames = []

    @property
    def data(self) -> pd.DataFrame | None:
        """All tracked detections as a DataFrame, or ``None`` until complete."""
        return self._data

    def boxes_for_frame(self, frame_index: int) -> list[BoundingBox]:
        """Object bounding boxes for frame ``frame_index`` (0-based)."""
        if self._data is None:
            return []
        positions = self._rows_by_frame.get(frame_index)
        if positions is None:
            return []
        rows = self._data.take(positions)
        return [
            BoundingBox(r.x1, r.y1, r.x2, r.y2, int(r.track_id))
            for r in rows.itertuples(index=False)
        ]

    def clear(self) -> None:
        self._data = None
        self._rows_by_frame = {}
        self._frames = []
