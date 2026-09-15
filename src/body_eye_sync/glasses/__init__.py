"""Read gaze and motion from Tobii Pro Glasses 2/3 folders or TSV gaze exports."""

from __future__ import annotations

import os
from pathlib import Path

from body_eye_sync.glasses import glasses2, glasses3, tsv
from body_eye_sync.glasses.data import (
    MotionData,
    Recording,
    Series,
    Streams,
    TrackingData,
)

__all__ = [
    "MotionData",
    "Recording",
    "Series",
    "Streams",
    "TrackingData",
    "find_recordings",
    "read_motion",
    "read_streams",
    "read_tracking",
    "recording_folder",
    "recording_for_video",
    "recording_info",
]

#: The folder readers, tried in turn against a folder.
_READERS = (glasses3, glasses2)


def _reader_for(folder: Path):
    """Return the matching folder reader, or ``None``."""
    return next((reader for reader in _READERS if reader.is_recording(folder)), None)


#: Maximum recording search depth, including nested card layouts.
SEARCH_DEPTH = 5


def find_recordings(path: str | Path, *, depth: int = SEARCH_DEPTH) -> list[Path]:
    """Find recording roots in sorted traversal order, up to ``depth`` levels deep.

    Skip hidden subfolders and stop descending at recording roots.
    """
    path = Path(path)
    return _find_recordings(path, depth) if path.is_dir() else []


def _find_recordings(folder: Path, depth: int) -> list[Path]:
    if _reader_for(folder) is not None:
        return [folder]
    if depth <= 0:
        return []
    try:
        with os.scandir(folder) as entries:
            children = sorted(
                entry.name
                for entry in entries
                if entry.is_dir() and not entry.name.startswith(".")
            )
    except OSError:  # a folder we are not allowed to look inside holds nothing
        return []
    return [
        found
        for name in children
        for found in _find_recordings(folder / name, depth - 1)
    ]


def recording_for_video(path: str | Path) -> Path | None:
    """The recording root for a video in a supported device folder layout."""
    video = Path(path)
    for reader in _READERS:
        folder = reader.folder_for_video(video)
        if folder is not None:
            return folder
    return None


def recording_info(path: str | Path) -> Recording | None:
    """Read metadata without samples; ``None`` unless exactly one recording is found."""
    folder = recording_folder(path)
    reader = None if folder is None else _reader_for(folder)
    return None if reader is None else reader.info(folder)


def recording_folder(path: str | Path) -> Path | None:
    """Return the sole recording root at or below ``path``, or ``None``."""
    found = find_recordings(path)
    return found[0] if len(found) == 1 else None


def read_streams(path: str | Path, *, video_path: str | Path | None = None) -> Streams:
    """Read gaze and motion together from a recording folder or TSV export.

    TSV exports require ``video_path`` and return empty motion.
    """
    path = Path(path)
    folder = recording_folder(path)
    if folder is not None:
        return _reader_for(folder).read(folder)
    if path.is_file():
        return tsv.read(path, video_path=video_path)
    if path.is_dir():
        raise ValueError(
            f"{path} is not a glasses recording folder: expected "
            f"{glasses3.MANIFEST} (Glasses 3) or {glasses2.MANIFEST} with a "
            f"{glasses2.SEGMENTS} folder (Glasses 2)"
        )
    raise FileNotFoundError(f"no glasses recording at {path}")


def read_tracking(
    path: str | Path, *, video_path: str | Path | None = None
) -> TrackingData:
    """The eye tracking of a recording folder or a gaze export."""
    return read_streams(path, video_path=video_path).tracking


def read_motion(path: str | Path) -> MotionData:
    """Read head movement from a recording folder."""
    return read_streams(path).motion
