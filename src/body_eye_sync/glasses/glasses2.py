"""Read single-segment Tobii Pro Glasses 2 recordings.

``livedata.json.gz`` interleaves gaze and motion. Gaze records share a
``gidx``; nonzero ``s`` marks invalid readings. ``vts`` maps the device
microsecond clock to video time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from body_eye_sync.glasses.data import (
    MotionData,
    Recording,
    Series,
    Streams,
    TrackingData,
)
from body_eye_sync.glasses.frames import GLASSES2_IMU, to_head_unit
from body_eye_sync.glasses.records import json_file, json_lines
from body_eye_sync.media import VideoInfo, require_video_info, video_info

#: Names a folder as a Glasses 2 recording, together with ``SEGMENTS``.
MANIFEST = "recording.json"
SEGMENTS = "segments"
LIVE_DATA = "livedata.json.gz"
VIDEO = "fullstream.mp4"

DEVICE = "Tobii Pro Glasses 2"

#: Where each eye sits on the per-eye axis of the tracking arrays.
_EYE_INDEX = {"left": 0, "right": 1}


class _Readings:
    """Collect sparse readings by gaze index for vectorised array assembly."""

    def __init__(self, *, per_eye: bool = True) -> None:
        #: Whether a reading is of one eye, rather than of both together.
        self.per_eye = per_eye
        self.samples: list[int] = []
        self.eyes: list[int] = []
        self.values: list[float | list[float]] = []

    def add(self, sample: int, value: float | list[float], eye: int = 0) -> None:
        """Store a reading for a sample and optional eye."""
        self.samples.append(sample)
        self.eyes.append(eye)
        self.values.append(value)

    def scatter(self, rows: dict[int, int], shape: tuple[int, ...]) -> np.ndarray:
        """Scatter readings using the sample-to-row mapping.

        Leave missing values as NaN and discard samples absent from ``rows``.
        """
        array = np.full(shape, np.nan)
        if not self.samples:
            return array
        rows_of = np.array([rows.get(sample, -1) for sample in self.samples])
        kept = rows_of >= 0
        index: tuple[np.ndarray, ...] = (rows_of[kept],)
        if self.per_eye:
            index += (np.array(self.eyes)[kept],)
        array[index] = np.array(self.values, dtype=float)[kept]
        return array


def is_recording(folder: Path) -> bool:
    """Whether ``folder`` is a Glasses 2 recording folder."""
    return (folder / MANIFEST).is_file() and (folder / SEGMENTS).is_dir()


def _segment(folder: Path) -> Path:
    """Return the sole numeric segment directory; reject zero or multiple segments."""
    segments = [
        path
        for path in (folder / SEGMENTS).iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not segments:
        raise FileNotFoundError(f"no recording segments in {folder / SEGMENTS}")
    if len(segments) > 1:
        raise ValueError(
            f"{folder.name} was recorded in {len(segments)} segments, each with its "
            "own video; only single-segment recordings can be imported"
        )
    return segments[0]


def video_path(folder: Path) -> Path:
    """Return the scene video path, whether or not the file exists."""
    return _segment(Path(folder)) / VIDEO


def folder_for_video(video: Path) -> Path | None:
    """The recording folder ``video`` is the scene video of, if any."""
    segment = video.parent
    if not (segment.name.isdigit() and segment.parent.name == SEGMENTS):
        return None
    folder = segment.parent.parent
    return folder if is_recording(folder) and video_path(folder) == video else None


def _video_clock_offset(vts_records: list[tuple[int, int]], source: Path) -> int:
    """Median device timestamp of the first video frame, in microseconds.

    Raise ``ValueError`` if there are no ``vts`` records.
    """
    if not vts_records:
        raise ValueError(
            f"{source} has no vts records, so its eye tracking cannot be placed "
            "against its video"
        )
    return int(
        np.median([timestamp - video_time for timestamp, video_time in vts_records])
    )


def info(folder: str | Path) -> Recording:
    """Read recording metadata without sensor samples."""
    folder = Path(folder)
    video = video_path(folder)
    return _info(folder, video, video_info(video))


def _info(folder: Path, video: Path, metadata: VideoInfo) -> Recording:
    """Read recording metadata using an already-resolved video and its header."""
    participant = json_file(folder / "participant.json").get("pa_info") or {}
    return Recording(
        source=folder,
        device=DEVICE,
        video_path=video,
        resolution=metadata.size,
        participant=participant.get("Name") or None,
        serial=json_file(folder / "sysinfo.json").get("hu_serial"),
    )


def read(folder: str | Path) -> Streams:
    """Read gaze and motion in one pass, converting times to the container clock."""
    folder = Path(folder)
    segment = _segment(folder)
    live_data = segment / LIVE_DATA
    if not live_data.is_file():
        raise FileNotFoundError(f"no sensor data in {segment}: {LIVE_DATA} is missing")

    timestamps: list[int] = []
    gaze_records: list[list[float]] = []
    rows: dict[int, int] = {}
    vts_records: list[tuple[int, int]] = []
    acceleration: list[tuple[int, list[float]]] = []
    rotation: list[tuple[int, list[float]]] = []
    gaze_3d = _Readings(per_eye=False)
    pupil = _Readings()
    origin = _Readings()
    direction = _Readings()

    for record in json_lines(live_data):
        if "vts" in record:
            vts_records.append((record["ts"], record["vts"]))
            continue
        if "ac" in record:
            acceleration.append(
                (record["ts"], record["ac"] if record["s"] == 0 else [np.nan] * 3)
            )
            continue
        if "gy" in record:
            rotation.append(
                (record["ts"], record["gy"] if record["s"] == 0 else [np.nan] * 3)
            )
            continue
        index = record.get("gidx")
        if index is None:
            continue  # a sync port event, or the video's own timestamps
        valid = record["s"] == 0
        if "gp" in record:
            rows[index] = len(timestamps)
            timestamps.append(record["ts"])
            gaze_records.append(record["gp"] if valid else [np.nan, np.nan])
        elif "gp3" in record:
            if valid:
                gaze_3d.add(index, record["gp3"])
        elif (eye := _EYE_INDEX.get(record.get("eye"))) is not None and valid:
            if "pd" in record:
                pupil.add(index, record["pd"], eye)
            elif "pc" in record:
                origin.add(index, record["pc"], eye)
            elif "gd" in record:
                direction.add(index, record["gd"], eye)

    count = len(timestamps)
    offset = _video_clock_offset(vts_records, live_data)
    video = segment / VIDEO
    metadata = require_video_info(video, needed_by=folder)
    recording = _info(folder, video, metadata)

    def to_video_clock(stamps: list[int]) -> np.ndarray:
        """Device microseconds to seconds on the video container clock."""
        return (np.array(stamps, dtype=float) - offset) / 1e6 + metadata.start

    def series(readings: list[tuple[int, list[float]]]) -> Series:
        """Convert sensor timestamps and axes to the shared conventions."""
        return Series(
            time=to_video_clock([stamp for stamp, _ in readings]),
            values=to_head_unit([value for _, value in readings], GLASSES2_IMU),
        )

    tracking = TrackingData(
        time=to_video_clock(timestamps),
        gaze=np.array(gaze_records, dtype=float).reshape(-1, 2),
        pupil=pupil.scatter(rows, (count, 2)),
        gaze_3d=gaze_3d.scatter(rows, (count, 3)),
        eye_origin=origin.scatter(rows, (count, 2, 3)),
        eye_direction=direction.scatter(rows, (count, 2, 3)),
        recording=recording,
    )
    return Streams(
        tracking=tracking,
        motion=MotionData(
            acceleration=series(acceleration),
            angular_velocity=series(rotation),
            recording=recording,
        ),
    )
