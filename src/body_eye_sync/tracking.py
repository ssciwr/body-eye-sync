"""Person tracking using BoxMOT."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NamedTuple

# Hardcoded tracking parameters. These keep the public function simple for now;
# they can be promoted to arguments once the pipeline needs to vary them.
_DETECTOR = "yolov8n"
_REID = "osnet_x0_25_msmt17"
_TRACKER = "botsort"
_DEVICE = "cpu"


class Detection(NamedTuple):
    """A single tracked detection within a tracklet."""

    frame: int
    x: float
    y: float
    w: float
    h: float
    conf: float


def detect_tracklets(video_path: str | Path) -> dict[int, list[Detection]]:
    """Track people in a video and return the detections grouped by tracklet.

    Runs BoxMOT (person detection + ReID tracking) over every frame of
    ``video_path`` and returns a mapping from track id to the list of
    detections making up that tracklet, ordered by frame.
    """
    from boxmot import Boxmot

    with tempfile.TemporaryDirectory() as workspace:
        runner = Boxmot(
            detector=_DETECTOR,
            reid=_REID,
            tracker=_TRACKER,
            classes=[0],  # COCO class 0 == person
            project=workspace,
        )
        result = runner.track(
            source=str(video_path),
            device=_DEVICE,
            save_txt=True,
            verbose=False,
        )
        return _parse_mot(Path(result.text_path))


def _parse_mot(path: Path) -> dict[int, list[Detection]]:
    """Parse a BoxMOT MOT results file (``frame,id,x,y,w,h,conf,cls,det_ind``)."""
    tracklets: dict[int, list[Detection]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        frame, track_id, x, y, w, h, conf = line.split(",")[:7]
        tracklets.setdefault(int(track_id), []).append(
            Detection(int(frame), float(x), float(y), float(w), float(h), float(conf))
        )
    for detections in tracklets.values():
        detections.sort(key=lambda d: d.frame)
    return tracklets
