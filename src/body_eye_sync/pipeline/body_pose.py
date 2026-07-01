"""Body pose detection inside tracked person boxes using Ultralytics YOLO pose."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from body_eye_sync.pipeline.object_tracking import (
    BoundingBox,
    cached_model_path,
    default_device,
)

#: Default Ultralytics pose weights. Resolved into the app cache before loading.
DEFAULT_MODEL_NAME = "yolov8n-pose.pt"

#: The COCO keypoints returned by Ultralytics YOLO pose models.
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

#: COCO skeleton edges, as pairs of indices into :data:`KEYPOINT_NAMES`.
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

#: Per-keypoint coordinate/confidence columns, e.g. ``pose_nose_x``.
_KEYPOINT_COLUMNS = [
    f"pose_{name}_{field}" for name in KEYPOINT_NAMES for field in ("x", "y", "score")
]

#: The columns body-pose detection contributes, keyed onto ``(frame, track_id)``.
POSE_COLUMNS = [
    "pose_score",
    "pose_x1",
    "pose_y1",
    "pose_x2",
    "pose_y2",
    *_KEYPOINT_COLUMNS,
]

#: Columns of the per-frame pose DataFrame before it is merged onto the tracks.
_COLUMNS = ["frame", "track_id", *POSE_COLUMNS]


@dataclass
class BodyPose:
    """A single detected body pose for one tracked person box.

    ``box`` is the YOLO pose bounding box, in video-pixel coordinates, carrying
    the tracked person's ``track_id``. ``keypoints`` holds ``(x, y, score)``
    triples in :data:`KEYPOINT_NAMES` order.
    """

    box: BoundingBox
    score: float
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class PoseFrameResult:
    """Body poses detected in one video frame, with a 0-based ``frame_idx``."""

    frame_idx: int
    poses: list[BodyPose]


def poses_to_dataframe(frames: Iterable[PoseFrameResult]) -> pd.DataFrame:
    """Stack per-frame pose results into a DataFrame keyed on ``(frame, track_id)``.

    Returns a :class:`pandas.DataFrame` with columns ``frame, track_id`` plus
    :data:`POSE_COLUMNS`, ready to left-merge onto the stored tracks. Frames with
    no detected poses contribute no rows.
    """
    rows = []
    for result in frames:
        for pose in result.poses:
            row = [
                result.frame_idx,
                pose.box.track_id,
                pose.score,
                pose.box.x1,
                pose.box.y1,
                pose.box.x2,
                pose.box.y2,
            ]
            for px, py, score in pose.keypoints:
                row.extend((px, py, score))
            rows.append(row)
    data = np.asarray(rows, dtype=float) if rows else np.empty((0, len(_COLUMNS)))
    return pd.DataFrame(data, columns=_COLUMNS).astype({"frame": int, "track_id": int})


def pose_from_row(row) -> BodyPose:
    """Rebuild a drawable :class:`BodyPose` from a merged DataFrame row."""
    keypoints = [
        (
            float(getattr(row, f"pose_{name}_x")),
            float(getattr(row, f"pose_{name}_y")),
            float(getattr(row, f"pose_{name}_score")),
        )
        for name in KEYPOINT_NAMES
    ]
    box = BoundingBox(
        float(row.pose_x1),
        float(row.pose_y1),
        float(row.pose_x2),
        float(row.pose_y2),
        int(row.track_id),
    )
    return BodyPose(box, float(row.pose_score), keypoints)


def _clamp_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Round a box to integer pixels and clip it to the frame bounds."""
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


def _as_array(value) -> np.ndarray:
    """Convert Ultralytics/torch/numpy values to a float numpy array."""
    if value is None:
        return np.empty(0, dtype=float)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _detect_in_boxes(
    model,
    image,
    boxes: Sequence[BoundingBox],
    width: int,
    height: int,
    conf: float,
    device: str,
) -> list[BodyPose]:
    """Best body pose (if any) for each person box in one frame, in frame pixels."""
    poses: list[BodyPose] = []
    for person in boxes:
        x0, y0, x1c, y1c = _clamp_box(
            person.x1, person.y1, person.x2, person.y2, width, height
        )
        if x1c - x0 < 8 or y1c - y0 < 8:
            continue
        crop = image[y0:y1c, x0:x1c]
        result = model.predict(crop, conf=conf, device=device, verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            continue

        xyxy = _as_array(result.boxes.xyxy)
        scores = _as_array(result.boxes.conf).reshape(-1)
        keypoint_xy = _as_array(result.keypoints.xy)
        keypoint_scores = _as_array(result.keypoints.conf)
        if len(xyxy) == 0 or len(scores) == 0 or len(keypoint_xy) == 0:
            continue

        if keypoint_scores.size == 0:
            keypoint_scores = np.ones(keypoint_xy.shape[:2], dtype=float)

        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(np.argmax(scores * areas))
        px1, py1, px2, py2 = (float(v) for v in xyxy[best_idx])
        keypoints = [
            (x0 + float(px), y0 + float(py), float(score))
            for (px, py), score in zip(keypoint_xy[best_idx], keypoint_scores[best_idx])
        ]
        box = BoundingBox(x0 + px1, y0 + py1, x0 + px2, y0 + py2, person.track_id)
        poses.append(BodyPose(box, float(scores[best_idx]), keypoints))
    return poses


def detect_body_poses(
    video_path: str | Path,
    boxes_by_frame: Mapping[int, Sequence[BoundingBox]],
    model_name: str | Path = DEFAULT_MODEL_NAME,
    conf: float = 0.25,
    device: str | None = None,
) -> Iterator[PoseFrameResult]:
    """Detect one body pose per tracked person box, yielding a result per frame.

    ``boxes_by_frame`` maps a 0-based frame index to that frame's tracked person
    :class:`BoundingBox`es. For each requested frame, each person crop is passed
    through an Ultralytics YOLO pose model; the highest scoring pose in the crop
    is kept, its bounding box and COCO keypoints are offset back into full-frame
    coordinates and tagged with the person's ``track_id``. Every requested frame
    is yielded, including frames where no pose cleared ``conf``.
    """
    import cv2
    from ultralytics import YOLO

    if device is None:
        device = default_device()

    model = YOLO(cached_model_path(model_name))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        if not boxes_by_frame:
            return
        wanted = set(boxes_by_frame)
        last_frame = max(wanted)
        frame_idx = -1
        while frame_idx < last_frame:
            if not capture.grab():
                break
            frame_idx += 1
            if frame_idx not in wanted:
                continue
            ok, image = capture.retrieve()
            if not ok:
                break
            poses = _detect_in_boxes(
                model, image, boxes_by_frame[frame_idx], width, height, conf, device
            )
            yield PoseFrameResult(frame_idx, poses)
    finally:
        capture.release()
