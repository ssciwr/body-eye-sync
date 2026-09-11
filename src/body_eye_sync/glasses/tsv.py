"""Read TSV gaze exports, normalising pixel coordinates and converting timestamps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from body_eye_sync.glasses.data import (
    MotionData,
    Recording,
    Streams,
    TrackingData,
)
from body_eye_sync.media import require_video_info

DEVICE = "gaze export"

TIME = "gaze_video_time"
RECORDED = "rectimestamp"
GAZE = ("gaze_x", "gaze_y")
PUPIL = ("pupil_left", "pupil_right")
PARTICIPANT = "participant"

#: Columns without which the file cannot be read as gaze samples.
REQUIRED = (TIME, *GAZE, *PUPIL)

#: Pupil diameters are exported as thousandths of a millimetre.
_PUPIL_SCALE = 1e-3


def _times(samples: pd.DataFrame) -> np.ndarray:
    """Sample times in seconds relative to the first video frame.

    Align ``rectimestamp`` using its median offset from ``gaze_video_time``
    to avoid 10 ms rounding. Fall back to ``gaze_video_time`` if unavailable.
    """
    video_time = samples[TIME].to_numpy(dtype=float) / 1e3
    if RECORDED not in samples.columns or len(samples) < 2:
        return video_time
    recorded = samples[RECORDED].to_numpy(dtype=float) / 1e3
    return recorded - float(np.median(recorded - video_time))


def read(path: str | Path, *, video_path: str | Path | None = None) -> Streams:
    """Read a gaze export against the scene video its pixels were measured in.

    Return normalised gaze on the container clock and empty motion data.
    """
    path = Path(path)
    if video_path is None:
        raise ValueError(
            f"{path.name} holds gaze in video pixels timed by that video, so it "
            "can only be read alongside it; pass video_path"
        )
    video_path = Path(video_path)
    video = require_video_info(video_path, needed_by=path)

    samples = pd.read_csv(path, sep="\t")
    missing = [column for column in REQUIRED if column not in samples.columns]
    if missing:
        raise ValueError(f"{path.name} has no {', '.join(missing)} column")

    gaze = samples[list(GAZE)].to_numpy(dtype=float) / np.asarray(video.size, float)
    pupil = samples[list(PUPIL)].to_numpy(dtype=float) * _PUPIL_SCALE
    # Treat zero placeholders as missing, like the NA values parsed above.
    gaze[(gaze == 0.0).all(axis=1)] = np.nan
    pupil[pupil == 0.0] = np.nan

    participant = None
    if PARTICIPANT in samples.columns and samples[PARTICIPANT].nunique() == 1:
        participant = str(samples[PARTICIPANT].iloc[0])
    recording = Recording(
        source=path,
        device=DEVICE,
        video_path=video_path,
        resolution=video.size,
        participant=participant,
    )
    tracking = TrackingData(
        # Video time counts from the first frame; shift to the container clock.
        time=_times(samples) + video.start,
        gaze=gaze,
        pupil=pupil,
        recording=recording,
    )
    return Streams(tracking=tracking, motion=MotionData.empty(recording))
