"""Face detection inside tracked person boxes using InsightFace"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from body_eye_sync.pipeline.object_tracking import BoundingBox

#: The five InsightFace keypoints, in the order they appear in ``face.kps``.
LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]

#: Per-landmark coordinate columns, e.g. ``left_eye_x, left_eye_y``.
_LANDMARK_COLUMNS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]

#: The columns face detection contributes, keyed onto ``(frame, track_id)``.
FACE_COLUMNS = [
    "face_score",
    "face_x1",
    "face_y1",
    "face_x2",
    "face_y2",
    *_LANDMARK_COLUMNS,
]

#: Columns of the per-frame face DataFrame before it is merged onto the tracks.
_COLUMNS = ["frame", "track_id", *FACE_COLUMNS]


@dataclass
class FaceBox:
    """A single detected face: its bounding box plus face-specific attributes.

    ``box`` is the face's bounding box, in video-pixel coordinates, carrying the
    tracked person's ``track_id``. ``landmarks`` holds the five ``(x, y)``
    keypoints in :data:`LANDMARK_NAMES` order.
    """

    box: BoundingBox
    score: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FaceFrameResult:
    """Faces detected in one video frame, with a 0-based ``frame_idx``."""

    frame_idx: int
    faces: list[FaceBox]


def faces_to_dataframe(frames: Iterable[FaceFrameResult]) -> pd.DataFrame:
    """Stack per-frame face results into a DataFrame keyed on ``(frame, track_id)``.

    Returns a :class:`pandas.DataFrame` with columns ``frame, track_id`` plus
    :data:`FACE_COLUMNS`, ready to left-merge onto the stored tracks. Frames with
    no detected faces contribute no rows.
    """
    rows = []
    for result in frames:
        for face in result.faces:
            row = [
                result.frame_idx,
                face.box.track_id,
                face.score,
                face.box.x1,
                face.box.y1,
                face.box.x2,
                face.box.y2,
            ]
            for px, py in face.landmarks:
                row.extend((px, py))
            rows.append(row)
    data = np.asarray(rows, dtype=float) if rows else np.empty((0, len(_COLUMNS)))
    return pd.DataFrame(data, columns=_COLUMNS).astype({"frame": int, "track_id": int})


def face_box_from_row(row) -> FaceBox:
    """Rebuild a drawable :class:`FaceBox` from a merged DataFrame row."""
    landmarks = [
        (float(getattr(row, f"{name}_x")), float(getattr(row, f"{name}_y")))
        for name in LANDMARK_NAMES
    ]
    box = BoundingBox(
        float(row.face_x1),
        float(row.face_y1),
        float(row.face_x2),
        float(row.face_y2),
        int(row.track_id),
    )
    return FaceBox(box, float(row.face_score), landmarks)


def default_providers() -> list[str]:
    """Pick onnxruntime execution providers, preferring CUDA when available."""
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


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


def _detect_in_boxes(
    app, image, boxes: Sequence[BoundingBox], width: int, height: int, det_thresh: float
) -> list[FaceBox]:
    """Best face (if any) for each person box in one frame, in frame pixels."""
    faces: list[FaceBox] = []
    for person in boxes:
        x0, y0, x1c, y1c = _clamp_box(
            person.x1, person.y1, person.x2, person.y2, width, height
        )
        if x1c - x0 < 8 or y1c - y0 < 8:
            continue
        crop = image[y0:y1c, x0:x1c]
        detected = app.get(crop)
        if not detected:
            continue
        best = max(
            detected,
            key=lambda f: (
                float(f.det_score) * (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ),
        )
        if float(best.det_score) < det_thresh:
            continue
        fx1, fy1, fx2, fy2 = (float(v) for v in best.bbox)
        landmarks = [(x0 + float(px), y0 + float(py)) for px, py in best.kps]
        box = BoundingBox(x0 + fx1, y0 + fy1, x0 + fx2, y0 + fy2, person.track_id)
        faces.append(FaceBox(box, float(best.det_score), landmarks))
    return faces


def detect_faces(
    video_path: str | Path,
    boxes_by_frame: Mapping[int, Sequence[BoundingBox]],
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_thresh: float = 0.5,
    providers: list[str] | None = None,
) -> Iterator[FaceFrameResult]:
    """Detect one face per tracked person box, yielding a result per frame.

    ``boxes_by_frame`` maps a 0-based frame index to that frame's tracked person
    :class:`BoundingBox`es, as produced by object tracking. For each such frame
    the person crop is run through InsightFace; the highest scoring face above
    ``det_thresh`` is kept, its bounding box and five landmarks offset back into
    full-frame coordinates and tagged with the box's ``track_id``. Every
    requested frame is yielded -- including frames where no face cleared the
    threshold (their ``faces`` is empty) -- so callers can show progress and
    cancel responsively. Stop iterating to cancel early.
    """
    import cv2
    from insightface.app import FaceAnalysis

    if providers is None:
        providers = default_providers()

    app = FaceAnalysis(name=model_name, providers=providers)
    ctx_id = 0 if providers[0].startswith("CUDA") else -1
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

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
        # Walk frames in order, only decoding (retrieve) the ones we need; this
        # avoids the per-frame seeks that random access would cost.
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
            faces = _detect_in_boxes(
                app, image, boxes_by_frame[frame_idx], width, height, det_thresh
            )
            yield FaceFrameResult(frame_idx, faces)
    finally:
        capture.release()
