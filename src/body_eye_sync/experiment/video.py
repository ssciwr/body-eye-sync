"""Object tracking and vision model outputs for a video."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from body_eye_sync.experiment.embeddings import (
    TopK,
    read_embeddings,
    write_embeddings,
)
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.glasses import MotionData, Streams, TrackingData, read_streams
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
from body_eye_sync.media import VideoInfo, video_info
from body_eye_sync.preprocessing.audio import has_audio_stream


#: Column layout of the embeddings table.
_EMBEDDING_COLUMNS = ["track_id", "frame", "score", "embedding"]


def _embeddings_filename(kind: str) -> str:
    """The file one kind of embedding is stored in, inside an output directory."""
    return f"{kind}_embeddings.parquet"


class Video:
    """A video input: its settings and the model outputs computed from it.

    ``id`` names the input and its output directory, and ``timeline`` places
    the video's own clock on the experiment clock.

    Completed results live in a single numeric :attr:`data` DataFrame. While a
    run is in progress, each frame's BoxMOT ``tracks`` array is accumulated and
    collapsed into that DataFrame once :meth:`finish_object_tracking` is called.
    Face detection runs as a later pass over those tracked boxes, accumulating
    per frame and folding its columns onto the matching rows in
    :meth:`finish_face_detection`. Body-pose detection follows the same pattern.

    A camera also records audio, so the speech stages can run over this video's
    own track, with their results stored in :attr:`speech`.
    """

    #: The tracked boxes, this input's main result.
    _RESULTS_FILENAME: ClassVar[str] = "results.parquet"

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        timeline: Timeline | None = None,
    ) -> None:
        self.id = id
        self.video_path = Path(path) if path is not None else None
        self.timeline = timeline if timeline is not None else Timeline()
        self.speech = Speech()
        self._has_audio_track = False
        self._audio_track_path: Path | None = None
        self._info: VideoInfo | None = None
        self._info_path: Path | None = None
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

    @property
    def path(self) -> Path | None:
        return self.video_path

    def _video_info(self) -> VideoInfo:
        """Cache video metadata until the path changes."""
        if self._info is None or self._info_path != self.video_path:
            self._info = (
                VideoInfo() if self.video_path is None else video_info(self.video_path)
            )
            self._info_path = self.video_path
        return self._info

    @property
    def fps(self) -> float:
        """Average frame rate, or zero if unavailable."""
        return self._video_info().frame_rate

    @property
    def video_start(self) -> float:
        """First frame timestamp in seconds on the container clock."""
        return self._video_info().start

    def _frame_rate(self) -> float:
        """Return a positive frame rate or raise ``ValueError``."""
        fps = self.fps
        if fps <= 0.0:
            raise ValueError(f"{self.id or 'video'} has no frame rate to count in")
        return fps

    def frame_at(self, local_time: float, *, nearest: bool = False) -> int:
        """Frame containing ``local_time`` on the container clock.

        With ``nearest=True``, select the nearest frame start instead.
        The index is not clamped to the video bounds.
        """
        frames = (local_time - self.video_start) * self._frame_rate()
        return round(frames) if nearest else math.floor(frames)

    def time_of_frame(self, index: int) -> float:
        """Frame start time in seconds on the container clock."""
        return self.video_start + index / self._frame_rate()

    def has_audio_track(self) -> bool:
        """Whether this video carries sound"""
        if self.video_path is None:
            return False
        if self._audio_track_path != self.video_path:
            self._has_audio_track = has_audio_stream(self.video_path)
            self._audio_track_path = self.video_path
        return self._has_audio_track

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
        self.speech.clear()
        self._tmp_frames = []
        self._tmp_face_frames = []
        self._tmp_pose_frames = []
        self._tmp_body_topk = TopK(0, _EMBEDDING_COLUMNS)
        self._tmp_face_topk = TopK(0, _EMBEDDING_COLUMNS)
        self._body_embeddings = None
        self._face_embeddings = None

    def has_data(self) -> bool:
        """Whether this video has any completed pipeline results in memory."""
        return self._data is not None or self.speech.data is not None

    def has_results(self, directory: str | Path) -> bool:
        """Whether ``directory`` already holds results for a video."""
        return (Path(directory) / self._RESULTS_FILENAME).exists()

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, one file per kind of result."""
        if self._data is None and self.speech.data is None:
            raise ValueError("no data to write; run the pipeline first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._data is not None:
            import pyarrow as pa
            import pyarrow.parquet as pq

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
        self.speech.save(directory)

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any."""
        directory = Path(directory)
        self.clear()
        self.speech.load(directory)
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


class GlassesVideo(Video):
    """Glasses video with gaze and motion loaded from a recording folder or TSV."""

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        gaze_path: str | Path | None = None,
        timeline: Timeline | None = None,
    ) -> None:
        super().__init__(id=id, path=path, timeline=timeline)
        self.gaze_path = Path(gaze_path) if gaze_path is not None else None
        self._streams: Streams | None = None

    def set_gaze(self, path: str | Path) -> None:
        """Change the gaze source and clear cached sensor data."""
        self.gaze_path = Path(path)
        self._streams = None

    @property
    def streams(self) -> Streams | None:
        """Lazily load and cache sensor data; ``None`` without a gaze source.

        Read errors propagate. :meth:`set_gaze` clears the cache.
        """
        if self._streams is None and self.gaze_path is not None:
            self._streams = read_streams(self.gaze_path, video_path=self.video_path)
        return self._streams

    @property
    def tracking(self) -> TrackingData | None:
        """Eye tracking, or ``None`` without a gaze source."""
        streams = self.streams
        return None if streams is None else streams.tracking

    @property
    def motion(self) -> MotionData | None:
        """Head motion; empty for TSV exports, ``None`` without a gaze source."""
        streams = self.streams
        return None if streams is None else streams.motion


class FixedVideo(Video):
    """Video from a camera at a fixed position in the room."""
