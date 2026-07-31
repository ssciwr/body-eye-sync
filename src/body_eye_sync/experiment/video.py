"""Object tracking and vision model outputs for a video."""

from __future__ import annotations

import heapq
import itertools
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


#: Column layout of the embeddings table.
_EMBEDDING_COLUMNS = ["track_id", "frame", "score", "embedding"]


class _TrackTopK:
    """Keeps the best-K embeddings per track id as ``float16``, ranked by score."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._heaps: dict[int, list] = {}
        self._counter = itertools.count()

    def add(self, track_id: int, frame: int, score: float, embedding) -> None:
        if self._k <= 0 or embedding is None:
            return
        vec = np.asarray(embedding, dtype=np.float16)
        # A per-track min-heap keyed by score keeps the K highest scoring; the
        # counter breaks score ties so the arrays are never compared.
        item = (float(score), next(self._counter), int(frame), vec)
        heap = self._heaps.setdefault(int(track_id), [])
        if len(heap) < self._k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    def to_frame(self) -> pd.DataFrame | None:
        """Best-first table of the kept embeddings, or ``None`` if none were kept."""
        rows = []
        for track_id, heap in self._heaps.items():
            for score, _, frame, vec in sorted(heap, key=lambda x: x[0], reverse=True):
                rows.append((track_id, frame, score, vec))
        if not rows:
            return None
        return pd.DataFrame(rows, columns=_EMBEDDING_COLUMNS)


def _embeddings_path(path: str | Path, kind: str) -> Path:
    """Companion embeddings file beside a main output."""
    path = Path(path)
    return path.with_name(f"{kind}_embeddings.parquet")


def _write_embeddings(path: Path, table: pd.DataFrame) -> None:
    """Write a best-K embeddings table as fixed-size ``float16`` vectors."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    matrix = np.stack(table["embedding"].to_numpy())
    _, dim = matrix.shape
    values = pa.array(matrix.reshape(-1), type=pa.float16())
    out = pa.table(
        {
            "track_id": pa.array(table["track_id"].to_numpy(), pa.int64()),
            "frame": pa.array(table["frame"].to_numpy(), pa.int64()),
            "score": pa.array(table["score"].to_numpy(), pa.float32()),
            "embedding": pa.FixedSizeListArray.from_arrays(values, dim),
        }
    )
    pq.write_table(out, str(path))


def _read_embeddings(path: Path) -> pd.DataFrame:
    """Load a companion embeddings file back into a table of ``float16`` vectors."""
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pandas()


class Video:
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

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        time_offset: float = 0.0,
    ) -> None:
        self.id = id
        self.video_path = Path(path) if path is not None else None
        self.time_offset = time_offset
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
        self._tmp_body_topk = _TrackTopK(0)
        self._tmp_face_topk = _TrackTopK(0)

    def set_video(self, path: str | Path) -> None:
        """Set the current video, invalidating any previous model outputs."""
        self.clear()
        self.video_path = Path(path)

    def begin_object_tracking(self, embeddings_per_track: int = 0) -> None:
        """Drop any previous model outputs.

        ``embeddings_per_track`` keeps that many best body-appearance (ReID)
        embeddings per tracklet, ranked by detection confidence, for later
        identity clustering; ``0`` keeps none.
        """
        self.clear()
        self._tmp_body_topk = _TrackTopK(embeddings_per_track)

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
        self._tmp_face_topk = _TrackTopK(embeddings_per_track)
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
        self._tmp_face_topk = _TrackTopK(0)
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

    @property
    def body_embeddings(self) -> pd.DataFrame | None:
        """Best-K body-appearance embeddings per tracklet, or ``None``."""
        return self._body_embeddings

    @property
    def face_embeddings(self) -> pd.DataFrame | None:
        """Best-K face embeddings per tracklet, or ``None``."""
        return self._face_embeddings

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
        self._tmp_body_topk = _TrackTopK(0)
        self._tmp_face_topk = _TrackTopK(0)
        self._body_embeddings = None
        self._face_embeddings = None

    def to_parquet(self, path: str | Path) -> None:
        """Write the completed :attr:`data` to a Parquet file.

        Any collected embeddings are written to companion files beside ``path``
        (``body_embeddings.parquet`` / ``face_embeddings.parquet``).
        Raises :class:`ValueError` if there is no completed data to write.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._data is None:
            raise ValueError("no data to write; run the pipeline first")
        table = pa.Table.from_pandas(self._data, preserve_index=False)
        pq.write_table(table, str(path))
        for kind, embeddings in (
            ("body", self._body_embeddings),
            ("face", self._face_embeddings),
        ):
            embeddings_path = _embeddings_path(path, kind)
            if embeddings is None:
                embeddings_path.unlink(missing_ok=True)
            else:
                _write_embeddings(embeddings_path, embeddings)

    def load_parquet(self, path: str | Path) -> None:
        """Load results written by :meth:`to_parquet` into this video.

        Replaces any current results with the table at ``path`` and its companion
        embeddings files (``body_embeddings.parquet`` /
        ``face_embeddings.parquet``), if present.
        """
        self.clear()
        self.set_data(pd.read_parquet(path))
        body_path = _embeddings_path(path, "body")
        if body_path.exists():
            self._body_embeddings = _read_embeddings(body_path)
        face_path = _embeddings_path(path, "face")
        if face_path.exists():
            self._face_embeddings = _read_embeddings(face_path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "Video":
        """A new :class:`Video` loaded from a Parquet file (see :meth:`load_parquet`)."""
        video = cls()
        video.load_parquet(path)
        return video


class GlassesVideo(Video):
    """Video and gaze data from a participant's glasses-mounted camera."""

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        gaze_path: str | Path | None = None,
        time_offset: float = 0.0,
    ) -> None:
        super().__init__(id, path, time_offset)
        self.gaze_path = Path(gaze_path) if gaze_path is not None else None

    def set_gaze(self, path: str | Path) -> None:
        """Set the gaze samples recorded with this video."""
        self.gaze_path = Path(path)


class FixedVideo(Video):
    """Video from a camera at a fixed position in the room."""
