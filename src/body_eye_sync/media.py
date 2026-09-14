"""Shared timestamp metadata for audio and video containers."""

from __future__ import annotations

import math
from pathlib import Path


def _stream_timestamp(stream) -> float | None:
    """A stream's absolute start timestamp, when it has one."""
    if stream.start_time is None or stream.time_base is None:
        return None
    value = float(stream.start_time * stream.time_base)
    return value if math.isfinite(value) else None


def container_origin(container) -> float | None:
    """The timestamp used as zero for every stream in a container."""
    import av

    if container.start_time is not None:
        value = float(container.start_time / av.time_base)
        if math.isfinite(value):
            return value
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
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if math.isfinite(duration) and duration > 0.0:
            return duration
    if container.duration is not None:
        import av

        duration = float(container.duration / av.time_base) - stream_start(
            container, stream
        )
        if math.isfinite(duration) and duration > 0.0:
            return duration
    return None


def media_duration(path: str | Path) -> float | None:
    """How long a recording runs, read without decoding it."""
    import av

    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
                if math.isfinite(duration) and duration > 0.0:
                    return duration
            ends = [
                stream_start(container, stream) + duration
                for stream in container.streams
                if (duration := stream_duration(container, stream)) is not None
            ]
            return max(ends, default=None)
    except Exception:
        return None
