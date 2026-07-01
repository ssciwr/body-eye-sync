"""Object tracking using BoxMOT"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from boxmot.engine.tracking.results import FrameResult

#: Columns of the stored tracklets DataFrame.
_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]

# BoxMOT packs each frame's tracks into an (n, 8) array with these columns:
# x1, y1, x2, y2, id, conf, cls, det_ind.
_XYXY = slice(0, 4)
_ID = 4
_CONF = 5


@dataclass
class BoundingBox:
    """A single tracked detection, in video-pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int


def boxes_from_tracks(tracks: np.ndarray) -> list[BoundingBox]:
    """Drawable boxes for one frame's BoxMOT ``tracks`` array."""
    return [
        BoundingBox(
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            int(row[_ID]),
        )
        for row in np.asarray(tracks)
    ]


def tracks_to_dataframe(frames: Iterable[tuple[int, np.ndarray]]) -> pd.DataFrame:
    """Stack ``(frame_idx, tracks)`` pairs into the stored tracklets DataFrame.

    Returns a :class:`pandas.DataFrame` with columns
    ``frame, track_id, x1, y1, x2, y2, conf``, ``frame`` and ``track_id`` as
    integers -- a compact numeric form to store and analyse. Frames with no
    detections contribute no rows.
    """
    blocks = []
    for frame_idx, tracks in frames:
        tracks = np.asarray(tracks)
        if len(tracks) == 0:
            continue
        block = np.empty((len(tracks), len(_COLUMNS)))
        block[:, 0] = frame_idx
        block[:, 1] = tracks[:, _ID]
        block[:, 2:6] = tracks[:, _XYXY]
        block[:, 6] = tracks[:, _CONF]
        blocks.append(block)
    data = np.vstack(blocks) if blocks else np.empty((0, len(_COLUMNS)))
    return pd.DataFrame(data, columns=_COLUMNS).astype({"frame": int, "track_id": int})


def default_device() -> str:
    """Pick the fastest available device as a BoxMOT/Ultralytics device string."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def detect_tracklets(
    video_path: str | Path,
    detector: str = "yolov8n",
    reid: str = "osnet_x0_25_msmt17",
    tracker: str = "botsort",
    device: str | None = None,
    object_classes: Sequence[int] = (0,),
) -> Iterator[FrameResult]:
    """Track objects in a video, yielding BoxMOT's per-frame result.

    Runs BoxMOT (object detection + ReID tracking) and yields each frame's
    :class:`~boxmot.engine.tracking.results.FrameResult` as soon as it is
    computed, so callers can display or accumulate it live. Each result exposes
    ``frame_idx`` (1-based) and ``tracks`` -- an ``(n, 8)`` array of
    ``x1, y1, x2, y2, id, conf, cls, det_ind``. Every frame is yielded, including
    frames with no detections (their ``tracks`` is empty). Stop iterating to
    cancel tracking early.

    ``object_classes`` are the COCO class ids to detect; the default ``(0,)`` is
    people only. ``device`` is a BoxMOT/Ultralytics device string (e.g. ``"0"``,
    ``"mps"``, ``"cpu"``); when ``None`` the fastest available device is chosen
    automatically. The detections sharing a track id form a tracklet, and
    ``conf`` is the per-frame confidence of the object detector.
    """
    from boxmot import track
    from boxmot.engine.workflows.support import (
        build_detector_from_spec,
        build_tracker_from_spec,
        build_tracker_with_reid_spec,
    )

    if device is None:
        device = default_device()

    detector_runtime = build_detector_from_spec(
        detector, classes=list(object_classes), device=device
    )
    tracker_runtime = build_tracker_from_spec(tracker, device=device)
    reid_runtime = build_tracker_with_reid_spec(
        tracker, tracker_runtime, reid, device=device
    )

    yield from track(
        str(video_path), detector_runtime, reid_runtime, tracker_runtime, verbose=False
    )
