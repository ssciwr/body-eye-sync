"""Object tracking and vision model outputs for a video."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from body_eye_sync.experiment.embeddings import (
    TopK,
    read_embeddings,
    write_embeddings,
)
from body_eye_sync.experiment.timeline import Timeline
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


#: Column layout of the embeddings table.
_EMBEDDING_COLUMNS = ["track_id", "frame", "score", "embedding"]


def _embeddings_filename(kind: str) -> str:
    """The file one kind of embedding is stored in, inside an output directory."""
    return f"{kind}_embeddings.parquet"


class Video(Timeline):
    """A video input: its settings and the model outputs computed from it.

    ``id`` names the input and its output directory, and ``time_offset`` is the
    seconds to add to this video's own clock to reach experiment time.

    Completed results live in a single numeric :attr:`data` DataFrame. While a
    run is in progress, each frame's BoxMOT ``tracks`` array is accumulated and
    collapsed into that DataFrame once :meth:`finish_object_tracking` is called.
    Face detection runs as a later pass over those tracked boxes, accumulating
    per frame and folding its columns onto the matching rows in
    :meth:`finish_face_detection`. Body-pose detection follows the same pattern.
    """

    #: The tracked boxes, this input's main result.
    _RESULTS_FILENAME: ClassVar[str] = "results.parquet"

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        **timeline,
    ) -> None:
        # ``timeline`` is where this input sits on the experiment clock; see
        # :class:`~body_eye_sync.experiment.timeline.Timeline`.
        super().__init__(**timeline)
        self.id = id
        self.video_path = Path(path) if path is not None else None
        # Persistent results.
        self._data: pd.DataFrame | None = None
        self._rows_by_frame: dict[int, np.ndarray] = {}
        self._body_embeddings: pd.DataFrame | None = None
        self._face_embeddings: pd.DataFrame | None = None
        # Per-pass scratch: accumulated while a pass runs, then collapsed into the
        # results above and reset. ``_tmp_`` marks them as transient.
        self._tmp_frames: list[tuple[int, np.ndarray]] = []
        self._tmp_face_frames: list[FaceFrameResult] = []
        self._tmp_pose_frames: list[PoseFrameResult] = []
        self._tmp_body_topk = TopK(0, _EMBEDDING_COLUMNS)
        self._tmp_face_topk = TopK(0, _EMBEDDING_COLUMNS)

    @classmethod
    def from_config(cls, spec, resolve) -> "Video":
        """This input as its stored form describes it.

        ``resolve`` turns a stored path into the one to read at runtime, which
        only the experiment holding the folder can do.
        """
        return cls(spec.id, resolve(spec.path), **cls.timeline_kwargs(spec))

    @property
    def path(self) -> Path | None:
        return self.video_path

    def begin_object_tracking(self, embeddings_per_track: int = 0) -> None:
        """Drop any previous model outputs.

        ``embeddings_per_track`` keeps that many best body-appearance (ReID)
        embeddings per tracklet, ranked by detection confidence, for later
        identity clustering; ``0`` keeps none.
        """
        self.clear()
        self._tmp_body_topk = TopK(embeddings_per_track, _EMBEDDING_COLUMNS)

    def add_object_tracking_frame(self, frame) -> None:
        """Accumulate a BoxMOT per-frame result, converting to 0-based indices"""
        tracks = np.asarray(frame.tracks)
        self._tmp_frames.append((frame.frame_idx - 1, tracks))
        self._collect_body_embeddings(frame.frame_idx - 1, tracks, frame)

    def _collect_body_embeddings(
        self, frame_index: int, tracks: np.ndarray, frame
    ) -> None:
        """Feed this frame's ReID embeddings (if any) into the per-track top-K."""
        embeddings = getattr(frame, "embeddings", None)
        if embeddings is None:
            return
        embeddings = np.asarray(embeddings)
        for row, vec in zip(tracks, embeddings):
            if not np.any(np.isfinite(vec)):
                continue  # predicted-only track with no detection this frame
            self._tmp_body_topk.add(int(row[4]), frame_index, float(row[5]), vec)

    def finish_object_tracking(self) -> None:
        """Collapse the streamed frames into the stored :attr:`data` DataFrame."""
        self.set_data(tracks_to_dataframe(self._tmp_frames))
        self._body_embeddings = self._tmp_body_topk.to_frame()

    def discard_object_tracking(self) -> None:
        """Drop a cancelled or failed run; its partial output is unusable."""
        self.clear()

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace all results with a complete data DataFrame."""
        if "frame" not in data.columns:
            raise ValueError("results table has no 'frame' column")
        self._data = data
        self._rows_by_frame = data.groupby("frame").indices
        self._tmp_frames = []

    def all_boxes_by_frame(self) -> dict[int, list[BoundingBox]]:
        """Tracked person boxes grouped by frame, as later passes consume them."""
        if self._data is None:
            return {}
        return {
            int(frame): self.boxes_for_frame(int(frame))
            for frame in self._rows_by_frame
        }

    def begin_face_detection(self, embeddings_per_track: int = 0) -> None:
        """Drop any previous face columns so a fresh pass starts clean.

        ``embeddings_per_track`` keeps that many best face embeddings per
        tracklet, ranked by face score, for later identity clustering.
        """
        if self._data is not None:
            present = [c for c in FACE_COLUMNS if c in self._data.columns]
            if present:
                self.set_data(self._data.drop(columns=present))
        self._tmp_face_frames = []
        self._tmp_face_topk = TopK(embeddings_per_track, _EMBEDDING_COLUMNS)
        self._face_embeddings = None

    def add_face_detection_frame(self, result: FaceFrameResult) -> None:
        """Accumulate one frame's detected faces for the final merge."""
        self._tmp_face_frames.append(result)
        for face in result.faces:
            self._tmp_face_topk.add(
                face.box.track_id, result.frame_idx, face.score, face.embedding
            )

    def finish_face_detection(self) -> None:
        """Merge the streamed faces onto their ``(frame, track_id)`` rows."""
        if self._data is None:
            return
        faces = faces_to_dataframe(self._tmp_face_frames)
        self.set_data(self._data.merge(faces, on=["frame", "track_id"], how="left"))
        self._tmp_face_frames = []
        self._face_embeddings = self._tmp_face_topk.to_frame()

    def discard_face_detection(self) -> None:
        """Drop a cancelled or failed pass; the tracked boxes are left intact."""
        self._tmp_face_frames = []
        self._tmp_face_topk = TopK(0, _EMBEDDING_COLUMNS)
        self._face_embeddings = None

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

    def begin_body_pose_detection(self, embeddings_per_track: int = 0) -> None:
        """Drop any previous pose columns so a fresh pass starts clean.

        ``embeddings_per_track`` is accepted for a uniform ``begin_*`` signature
        across steps but ignored -- pose detection produces no embeddings.
        """
        if self._data is not None:
            present = [c for c in POSE_COLUMNS if c in self._data.columns]
            if present:
                self.set_data(self._data.drop(columns=present))
        self._tmp_pose_frames = []

    def add_body_pose_frame(self, result: PoseFrameResult) -> None:
        """Accumulate one frame's detected body poses for the final merge."""
        self._tmp_pose_frames.append(result)

    def finish_body_pose_detection(self) -> None:
        """Merge the streamed poses onto their ``(frame, track_id)`` rows."""
        if self._data is None:
            return
        poses = poses_to_dataframe(self._tmp_pose_frames)
        self.set_data(self._data.merge(poses, on=["frame", "track_id"], how="left"))
        self._tmp_pose_frames = []

    def discard_body_pose_detection(self) -> None:
        """Drop a cancelled or failed pass; the tracked boxes are left intact."""
        self._tmp_pose_frames = []

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
        self._tmp_frames = []
        self._tmp_face_frames = []
        self._tmp_pose_frames = []
        self._tmp_body_topk = TopK(0, _EMBEDDING_COLUMNS)
        self._tmp_face_topk = TopK(0, _EMBEDDING_COLUMNS)
        self._body_embeddings = None
        self._face_embeddings = None

    def has_data(self) -> bool:
        """Whether this video has completed tracking results in memory."""
        return self._data is not None

    def has_results(self, directory: str | Path) -> bool:
        """Whether ``directory`` already holds results for a video."""
        return (Path(directory) / self._RESULTS_FILENAME).exists()

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, one file per kind of result."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._data is None:
            raise ValueError("no data to write; run the pipeline first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self._data, preserve_index=False)
        pq.write_table(table, str(directory / self._RESULTS_FILENAME))
        for kind, embeddings in (
            ("body", self._body_embeddings),
            ("face", self._face_embeddings),
        ):
            embeddings_path = directory / _embeddings_filename(kind)
            if embeddings is None:
                embeddings_path.unlink(missing_ok=True)
            else:
                write_embeddings(embeddings_path, embeddings)

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any."""
        directory = Path(directory)
        self.clear()
        results_path = directory / self._RESULTS_FILENAME
        if not results_path.exists():
            return
        self.set_data(pd.read_parquet(results_path))
        body_path = directory / _embeddings_filename("body")
        if body_path.exists():
            self._body_embeddings = read_embeddings(body_path)
        face_path = directory / _embeddings_filename("face")
        if face_path.exists():
            self._face_embeddings = read_embeddings(face_path)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Video":
        """A new :class:`Video` loaded from an output directory (see :meth:`load`)."""
        video = cls()
        video.load(directory)
        return video


class GlassesVideo(Video):
    """Video and gaze data from a participant's glasses-mounted camera."""

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        gaze_path: str | Path | None = None,
        **timeline,
    ) -> None:
        super().__init__(id, path, **timeline)
        self.gaze_path = Path(gaze_path) if gaze_path is not None else None

    @classmethod
    def from_config(cls, spec, resolve) -> "GlassesVideo":
        """As :meth:`Video.from_config`, and the gaze file recorded with it."""
        return cls(
            spec.id,
            resolve(spec.path),
            resolve(spec.gaze_path),
            **cls.timeline_kwargs(spec),
        )

    def set_gaze(self, path: str | Path) -> None:
        """Set the gaze samples recorded with this video."""
        self.gaze_path = Path(path)


class FixedVideo(Video):
    """Video from a camera at a fixed position in the room."""
