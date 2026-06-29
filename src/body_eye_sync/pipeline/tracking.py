"""Person tracking using BoxMOT."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from boxmot import track
from boxmot.engine.workflows.support import (
    build_detector_from_spec,
    build_tracker_from_spec,
    build_tracker_with_reid_spec,
)

if TYPE_CHECKING:
    from boxmot.engine.tracking.results import FrameResult


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
) -> Iterator[FrameResult]:
    """Track people in a video, yielding BoxMOT's per-frame result.

    Runs BoxMOT (person detection + ReID tracking) and yields each frame's
    :class:`~boxmot.engine.tracking.results.FrameResult` as soon as it is
    computed, so callers can display or accumulate it live. Each result exposes
    ``frame_idx`` (1-based) and ``tracks`` -- an ``(n, 8)`` array of
    ``x1, y1, x2, y2, id, conf, cls, det_ind``. Every frame is yielded, including
    frames with no detections (their ``tracks`` is empty). Stop iterating to
    cancel tracking early.

    ``device`` is a BoxMOT/Ultralytics device string (e.g. ``"0"``, ``"mps"``,
    ``"cpu"``); when ``None`` the fastest available device is chosen
    automatically. The detections sharing a track id form a tracklet, and
    ``conf`` is the per-frame confidence of the object detector.
    """
    if device is None:
        device = default_device()

    object_classes = [0]  # only detect people
    detector_runtime = build_detector_from_spec(
        detector, classes=object_classes, device=device
    )
    tracker_runtime = build_tracker_from_spec(tracker, device=device)
    reid_runtime = build_tracker_with_reid_spec(
        tracker, tracker_runtime, reid, device=device
    )

    yield from track(
        str(video_path), detector_runtime, reid_runtime, tracker_runtime, verbose=False
    )
