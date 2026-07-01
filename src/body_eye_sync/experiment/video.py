"""Object tracking and vision model outputs for a video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from body_eye_sync.pipeline.object_tracking import BoundingBox, tracks_to_dataframe
from body_eye_sync.pipeline.face_detection import (
    FACE_COLUMNS,
    FaceBox,
    FaceFrameResult,
    face_box_from_row,
    faces_to_dataframe,
)
from body_eye_sync.pipeline.body_pose import (
    POSE_COLUMNS,
    BodyPose,
    PoseFrameResult,
    pose_from_row,
    poses_to_dataframe,
)


class Video:
    """Contains object tracking and vision model outputs for a video.

    Completed results live in a single numeric :attr:`data` DataFrame. While a
    run is in progress, each frame's BoxMOT ``tracks`` array is accumulated and
    collapsed into that DataFrame once :meth:`finish_object_tracking` is called.
    Face detection runs as a later pass over those tracked boxes, accumulating
    per frame and folding its columns onto the matching rows in
    :meth:`finish_face_detection`. Body-pose detection follows the same pattern.
    """

    def __init__(self) -> None:
        self.video_path: Path | None = None
        self._data: pd.DataFrame | None = None
        self._rows_by_frame: dict[int, np.ndarray] = {}
        self._frames: list[tuple[int, np.ndarray]] = []
        self._face_frames: list[FaceFrameResult] = []
        self._pose_frames: list[PoseFrameResult] = []

    def set_video(self, path: str | Path) -> None:
        """Set the current video, invalidating any previous model outputs."""
        self.clear()
        self.video_path = Path(path)

    def begin_object_tracking(self) -> None:
        """Drop any previous model outputs."""
        self.clear()

    def add_object_tracking_frame(self, frame) -> None:
        """Accumulate a BoxMOT per-frame result, converting to 0-based indices"""
        self._frames.append((frame.frame_idx - 1, np.asarray(frame.tracks)))

    def finish_object_tracking(self) -> None:
        """Collapse the streamed frames into the stored :attr:`data` DataFrame."""
        self.set_data(tracks_to_dataframe(self._frames))

    def discard_object_tracking(self) -> None:
        """Drop a cancelled or failed run; its partial output is unusable."""
        self.clear()

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace all results with a complete data DataFrame."""
        self._data = data
        self._rows_by_frame = data.groupby("frame").indices
        self._frames = []

    def all_boxes_by_frame(self) -> dict[int, list[BoundingBox]]:
        """Tracked person boxes grouped by frame, as later passes consume them."""
        if self._data is None:
            return {}
        return {
            int(frame): self.boxes_for_frame(int(frame))
            for frame in self._rows_by_frame
        }

    def begin_face_detection(self) -> None:
        """Drop any previous face columns so a fresh pass starts clean."""
        if self._data is not None:
            present = [c for c in FACE_COLUMNS if c in self._data.columns]
            if present:
                self.set_data(self._data.drop(columns=present))
        self._face_frames = []

    def add_face_detection_frame(self, result: FaceFrameResult) -> None:
        """Accumulate one frame's detected faces for the final merge."""
        self._face_frames.append(result)

    def finish_face_detection(self) -> None:
        """Merge the streamed faces onto their ``(frame, track_id)`` rows."""
        if self._data is None:
            return
        faces = faces_to_dataframe(self._face_frames)
        self.set_data(self._data.merge(faces, on=["frame", "track_id"], how="left"))
        self._face_frames = []

    def discard_face_detection(self) -> None:
        """Drop a cancelled or failed pass; the tracked boxes are left intact."""
        self._face_frames = []

    def faces_for_frame(self, frame_index: int) -> list[FaceBox]:
        """Detected face boxes for frame ``frame_index`` (0-based)."""
        if self._data is None or "face_score" not in self._data.columns:
            return []
        positions = self._rows_by_frame.get(frame_index)
        if positions is None:
            return []
        rows = self._data.take(positions)
        rows = rows[rows["face_score"].notna()]
        return [face_box_from_row(r) for r in rows.itertuples(index=False)]

    def begin_body_pose_detection(self) -> None:
        """Drop any previous pose columns so a fresh pass starts clean."""
        if self._data is not None:
            present = [c for c in POSE_COLUMNS if c in self._data.columns]
            if present:
                self.set_data(self._data.drop(columns=present))
        self._pose_frames = []

    def add_body_pose_frame(self, result: PoseFrameResult) -> None:
        """Accumulate one frame's detected body poses for the final merge."""
        self._pose_frames.append(result)

    def finish_body_pose_detection(self) -> None:
        """Merge the streamed poses onto their ``(frame, track_id)`` rows."""
        if self._data is None:
            return
        poses = poses_to_dataframe(self._pose_frames)
        self.set_data(self._data.merge(poses, on=["frame", "track_id"], how="left"))
        self._pose_frames = []

    def discard_body_pose_detection(self) -> None:
        """Drop a cancelled or failed pass; the tracked boxes are left intact."""
        self._pose_frames = []

    def poses_for_frame(self, frame_index: int) -> list[BodyPose]:
        """Detected body poses for frame ``frame_index`` (0-based)."""
        if self._data is None or "pose_score" not in self._data.columns:
            return []
        positions = self._rows_by_frame.get(frame_index)
        if positions is None:
            return []
        rows = self._data.take(positions)
        rows = rows[rows["pose_score"].notna()]
        return [pose_from_row(r) for r in rows.itertuples(index=False)]

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
        self._face_frames = []
        self._pose_frames = []
