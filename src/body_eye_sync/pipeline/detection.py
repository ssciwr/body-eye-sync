"""The atoms of tracking output and their stored form.

Kept free of heavy dependencies (no boxmot/torch, no Qt) so it can be imported
cheaply by both the pipeline that produces detections and the app/GUI that
display and store them. The per-frame ``tracks`` arrays handled here use
BoxMOT's column layout, but no BoxMOT import is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

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
    object_id: int


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
