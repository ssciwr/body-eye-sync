"""Person tracking using BoxMOT."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from boxmot import track
from boxmot.engine.workflows.support import (
    build_detector_from_spec,
    build_tracker_from_spec,
    build_tracker_with_reid_spec,
)

_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]


def detect_tracklets(
    video_path: str | Path,
    detector: str = "yolov8n",
    reid: str = "osnet_x0_25_msmt17",
    tracker: str = "botsort",
    device: str = "cpu",
    progress: Callable[[int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Track people in a video and return one row per tracked detection.

    Runs BoxMOT (person detection + ReID tracking) over every frame of
    ``video_path``. Returns a :class:`pandas.DataFrame` with columns
    ``frame, track_id, x1, y1, x2, y2, conf``, sorted by frame; the detections
    sharing a ``track_id`` form a tracklet, and conf is the per-frame confidence
    of the object detector.

    If given, ``progress`` is called with the (1-based) frame index after each
    frame is processed, and ``is_cancelled`` is polled before each frame; when it
    returns ``True`` tracking stops early and the detections gathered so far are
    returned.
    """
    progress = progress or (lambda _frame_idx: None)
    is_cancelled = is_cancelled or (lambda: False)

    object_classes = [0]  # only detect people
    detector_runtime = build_detector_from_spec(
        detector, classes=object_classes, device=device
    )
    tracker_runtime = build_tracker_from_spec(tracker, device=device)
    reid_runtime = build_tracker_with_reid_spec(
        tracker, tracker_runtime, reid, device=device
    )

    blocks = []
    for frame_result in track(
        str(video_path), detector_runtime, reid_runtime, tracker_runtime, verbose=False
    ):
        if is_cancelled():
            break
        progress(frame_result.frame_idx)
        tracks = frame_result.tracks
        if len(tracks) == 0:
            continue
        blocks.append(
            np.column_stack(
                [
                    np.full(len(tracks), frame_result.frame_idx),
                    tracks.id,
                    tracks.xyxy,
                    tracks.conf,
                ]
            )
        )

    data = np.vstack(blocks) if blocks else np.empty((0, len(_COLUMNS)))
    df = pd.DataFrame(data, columns=_COLUMNS)
    return df.astype({"frame": int, "track_id": int})
