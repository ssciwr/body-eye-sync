"""Shared timestamp metadata for audio and video containers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0.0 else None


def _stream_seconds(ticks, stream) -> float | None:
    """A stream timestamp or duration in seconds, when the stream has one."""
    if ticks is None or stream.time_base is None:
        return None
    return _finite(float(ticks * stream.time_base))


def _container_seconds(ticks) -> float | None:
    """A container timestamp or duration in seconds, when the container has one."""
    import av

    return None if ticks is None else _finite(float(ticks / av.time_base))


def _stream_timestamp(stream) -> float | None:
    """A stream's absolute start timestamp, when it has one."""
    return _stream_seconds(stream.start_time, stream)


def container_origin(container) -> float | None:
    """The timestamp used as zero for every stream in a container."""
    origin = _container_seconds(container.start_time)
    if origin is not None:
        return origin
    starts = [
        value
        for stream in container.streams
        if (value := _stream_timestamp(stream)) is not None
    ]
    return min(starts, default=None)


def stream_start(container, stream) -> float:
    """A stream's start in seconds on its container's zero-based clock."""
    absolute = _stream_timestamp(stream)
    origin = container_origin(container)
    if absolute is None or origin is None:
        return 0.0
    return max(0.0, absolute - origin)


def stream_duration(container, stream) -> float | None:
    """A stream's duration, falling back to the remaining container span."""
    duration = _positive(_stream_seconds(stream.duration, stream))
    if duration is not None:
        return duration
    total = _container_seconds(container.duration)
    if total is None:
        return None
    return _positive(total - stream_start(container, stream))


def media_duration(path: str | Path) -> float | None:
    """How long a recording runs, read without decoding it."""
    import av

    try:
        with av.open(str(path)) as container:
            duration = _positive(_container_seconds(container.duration))
            if duration is not None:
                return duration
            ends = [
                stream_start(container, stream) + duration
                for stream in container.streams
                if (duration := stream_duration(container, stream)) is not None
            ]
            return max(ends, default=None)
    except Exception:
        return None


@dataclass(frozen=True)
class VideoInfo:
    """Video frame rate, start time, and dimensions from container metadata."""

    #: Average frames per second reported by the video stream.
    frame_rate: float = 0.0
    #: First frame timestamp in seconds on the container clock.
    start: float = 0.0
    #: Frame size, or ``None`` when there is no video stream to measure.
    size: tuple[int, int] | None = None


def video_info(path: str | Path) -> VideoInfo:
    """Read video metadata; return defaults for missing or unreadable video streams."""
    import av

    try:
        with av.open(str(path)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                return VideoInfo()
            rate = stream.average_rate
            return VideoInfo(
                frame_rate=float(rate) if rate else 0.0,
                start=stream_start(container, stream),
                size=(
                    (int(stream.width), int(stream.height))
                    if stream.width and stream.height
                    else None
                ),
            )
    except Exception:
        return VideoInfo()


def require_video_info(path: str | Path | None, *, needed_by: str | Path) -> VideoInfo:
    """Video metadata, raising unless there is a readable video to measure against.

    ``needed_by`` names the recording or export whose samples the video places.
    """
    if path is None:
        raise ValueError(f"{Path(needed_by).name} names no video to be read against")
    path = Path(path)
    if not path.is_file():
        problem = "is missing"
    elif (info := video_info(path)).size is None:
        problem = "holds no video stream"
    else:
        return info
    raise ValueError(
        f"{Path(needed_by).name} cannot be read without its video: "
        f"{path.name} {problem}"
    )
