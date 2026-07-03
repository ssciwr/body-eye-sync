"""Object tracking using BoxMOT"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from platformdirs import user_cache_path

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


def model_cache_dir() -> Path:
    """Directory all downloaded model weights are cached in (cross-platform)."""
    return user_cache_path("body-eye-sync", "SSC") / "models"


def cached_model_path(model_ref: str | Path) -> str:
    """Route a bare weights filename into the shared model cache.

    Given only a bare name (e.g. ``yolo26m.pt``), Ultralytics downloads the
    weights into the current working directory. Rewriting it to an absolute path
    under :func:`model_cache_dir` makes the download land in the shared cache
    instead. An explicit path (one with a directory part) or an already-existing
    file is returned unchanged.
    """
    path = Path(model_ref)
    if path.parent != Path(".") or path.exists():
        return str(model_ref)
    cache_dir = model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / path.name)


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
    # To get lazy, per-frame results so the GUI can display tracking live,
    # we can't use ``boxmot.Boxmot.track`` as this directly writes to a ``runs/`` directory.
    # So we use these internal ``build_*_from_spec`` imports and pin the boxmot version.
    from boxmot import track
    from boxmot.engine.workflows.support import (
        build_detector_from_spec,
        build_tracker_from_spec,
        build_tracker_with_reid_spec,
    )

    if device is None:
        device = default_device()

    # Pass an absolute cache path so the Ultralytics detector weights download
    # into the shared cache rather than the current working directory. (BoxMOT
    # already routes bare ReID names into its own weights dir.)
    detector_runtime = build_detector_from_spec(
        cached_model_path(detector), classes=list(object_classes), device=device
    )
    tracker_runtime = build_tracker_from_spec(tracker, device=device)
    reid_runtime = build_tracker_with_reid_spec(
        tracker, tracker_runtime, reid, device=device
    )

    yield from track(
        str(video_path), detector_runtime, reid_runtime, tracker_runtime, verbose=False
    )
