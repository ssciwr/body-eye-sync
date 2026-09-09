"""Read Tobii Pro Glasses 3 recordings described by ``recording.g3``.

Gaze and IMU timestamps count from the first scene camera frame, so they are
shifted onto the video container clock by that frame's start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
from packaging.version import InvalidVersion, Version

from body_eye_sync.glasses.data import (
    MotionData,
    Recording,
    Series,
    Streams,
    TrackingData,
)
from body_eye_sync.glasses.frames import GLASSES3_IMU, to_head_unit
from body_eye_sync.glasses.records import json_lines
from body_eye_sync.media import VideoInfo, require_video_info, video_info

#: The manifest, whose presence marks a folder as a Glasses 3 recording.
MANIFEST = "recording.g3"

DEVICE = "Tobii Pro Glasses 3"

#: First firmware version with IMU readings in head coordinates.
ALIGNED_FROM = Version("1.29")

_MISSING_3D = (np.nan, np.nan, np.nan)


def is_recording(folder: Path) -> bool:
    """Whether ``folder`` is a Glasses 3 recording folder."""
    return (folder / MANIFEST).is_file()


def _manifest(folder: Path) -> dict:
    return orjson.loads((folder / MANIFEST).read_bytes())


def video_path(folder: Path, manifest: dict) -> Path | None:
    """Return the manifest's scene video path, whether or not the file exists."""
    name = (manifest.get("scenecamera") or {}).get("file")
    return folder / name if name else None


def folder_for_video(video: Path) -> Path | None:
    """The recording folder ``video`` is the scene video of, if any."""
    folder = video.parent
    if not is_recording(folder):
        return None
    return folder if video_path(folder, _manifest(folder)) == video else None


def _meta(folder: Path, manifest: dict, name: str) -> str | None:
    """Read a named text file from the manifest's metadata folder."""
    path = folder / (manifest.get("meta-folder") or "meta") / name
    return path.read_text().strip() if path.is_file() else None


def _participant(folder: Path, manifest: dict) -> str | None:
    """Read the participant name from JSON or plain-text metadata."""
    raw = _meta(folder, manifest, "participant")
    if not raw:
        return None
    try:
        return orjson.loads(raw).get("name") or None
    except orjson.JSONDecodeError:
        return raw


def info(
    folder: str | Path,
    manifest: dict | None = None,
    metadata: VideoInfo | None = None,
) -> Recording:
    """Read recording metadata, reusing a parsed manifest and video header if given."""
    folder = Path(folder)
    manifest = _manifest(folder) if manifest is None else manifest
    scene = manifest.get("scenecamera") or {}
    video = video_path(folder, manifest)
    resolution = (scene.get("camera-calibration") or {}).get("resolution")
    if resolution is None and video is not None:
        # Metadata is read without the video, which may not be there to measure.
        resolution = (video_info(video) if metadata is None else metadata).size
    return Recording(
        source=folder,
        device=DEVICE,
        video_path=video,
        resolution=tuple(resolution) if resolution else None,
        participant=_participant(folder, manifest),
        serial=_meta(folder, manifest, "HuSerial"),
    )


def _imu_rotation(folder: Path, manifest: dict) -> np.ndarray | None:
    """Return the IMU correction for firmware before 1.29.

    Missing or invalid versions receive no correction.
    """
    version = _meta(folder, manifest, "RuVersion")
    if version is None:
        return None
    try:
        return GLASSES3_IMU if Version(version) < ALIGNED_FROM else None
    except InvalidVersion:
        return None


def _read_motion(
    folder: Path, manifest: dict, recording: Recording, video_start: float
) -> MotionData:
    """Read accelerometer and gyroscope samples; skip magnetometer records."""
    imu_file = folder / ((manifest.get("imu") or {}).get("file") or "imudata.gz")
    if not imu_file.is_file():
        return MotionData.empty(recording)
    times, acceleration, rotation = [], [], []
    for sample in json_lines(imu_file):
        data = sample.get("data") or {}
        if "accelerometer" not in data:
            continue  # a magnetometer reading, which we do not use
        times.append(sample["timestamp"])
        acceleration.append(data["accelerometer"])
        rotation.append(data["gyroscope"])
    turn = _imu_rotation(folder, manifest)
    time = np.array(times, dtype=float) + video_start
    return MotionData(
        acceleration=Series(time, to_head_unit(acceleration, turn)),
        angular_velocity=Series(time, to_head_unit(rotation, turn)),
        recording=recording,
    )


def read(folder: str | Path) -> Streams:
    """Read gaze and motion from a Glasses 3 recording."""
    folder = Path(folder)
    manifest = _manifest(folder)
    gaze_file = folder / ((manifest.get("gaze") or {}).get("file") or "gazedata.gz")
    if not gaze_file.is_file():
        raise FileNotFoundError(
            f"no gaze data in {folder}: {gaze_file.name} is missing"
        )

    times, gaze, gaze_3d, pupil, origin, direction = [], [], [], [], [], []
    for sample in json_lines(gaze_file):
        data = sample.get("data") or {}
        times.append(sample["timestamp"])
        # Invalid samples and untracked eyes are empty objects.
        gaze.append(data.get("gaze2d") or (np.nan, np.nan))
        gaze_3d.append(data.get("gaze3d") or _MISSING_3D)
        eyes = [data.get(f"eye{side}") or {} for side in ("left", "right")]
        pupil.append([eye.get("pupildiameter", np.nan) for eye in eyes])
        origin.append([eye.get("gazeorigin") or _MISSING_3D for eye in eyes])
        direction.append([eye.get("gazedirection") or _MISSING_3D for eye in eyes])

    metadata = require_video_info(video_path(folder, manifest), needed_by=folder)
    recording = info(folder, manifest, metadata)
    # Shift from first-frame time to the container clock.
    video_start = metadata.start
    tracking = TrackingData(
        time=np.array(times, dtype=float) + video_start,
        gaze=np.array(gaze, dtype=float).reshape(-1, 2),
        pupil=np.array(pupil, dtype=float).reshape(-1, 2),
        gaze_3d=np.array(gaze_3d, dtype=float).reshape(-1, 3),
        eye_origin=np.array(origin, dtype=float).reshape(-1, 2, 3),
        eye_direction=np.array(direction, dtype=float).reshape(-1, 2, 3),
        recording=recording,
    )
    return Streams(
        tracking=tracking,
        motion=_read_motion(folder, manifest, recording, video_start),
    )
