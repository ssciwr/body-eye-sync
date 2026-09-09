"""Shared data types for recording metadata, gaze, and head motion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


def _sample_rate(time: np.ndarray) -> float:
    """Samples per second from the median interval; zero with fewer than two."""
    if len(time) < 2:
        return 0.0
    return float(1.0 / np.median(np.diff(time)))


def _span(time: np.ndarray, start: float, end: float) -> slice:
    """The slice of sorted sample times in the half-open interval ``[start, end)``."""
    first, last = np.searchsorted(time, [start, end])
    return slice(int(first), int(last))


@dataclass(frozen=True)
class Recording:
    """Source, device, and video metadata."""

    #: The gaze export or recording folder the samples were read from.
    source: Path
    #: Device name, e.g. ``"Tobii Pro Glasses 3"``.
    device: str
    #: The scene video recorded alongside, when the source names one.
    video_path: Path | None = None
    #: Frame size of that video, needed to turn normalised gaze into pixels.
    resolution: tuple[int, int] | None = None
    #: Participant name from device metadata.
    participant: str | None = None
    #: Head unit serial number.
    serial: str | None = None


@dataclass(frozen=True)
class TrackingData:
    """Eye-tracking arrays with one row per sample, ordered by time.

    Times are seconds on the video container clock. Invalid readings are
    NaN; missing samples are not interpolated.
    """

    #: Seconds on the video container clock, one per sample.
    time: np.ndarray
    #: ``(n, 2)`` gaze point in the scene camera, normalised to ``0..1``.
    gaze: np.ndarray
    #: ``(n, 2)`` pupil diameter in mm, left eye then right.
    pupil: np.ndarray
    #: ``(n, 3)`` gaze point in mm in scene-camera space, when recorded.
    gaze_3d: np.ndarray | None = None
    #: ``(n, 2, 3)`` pupil centre in mm per eye, when recorded.
    eye_origin: np.ndarray | None = None
    #: ``(n, 2, 3)`` unit gaze direction per eye, when recorded.
    eye_direction: np.ndarray | None = None
    #: The source these samples were read from.
    recording: Recording | None = None

    def __post_init__(self) -> None:
        count = len(self.time)
        for name, expected in (
            ("gaze", (count, 2)),
            ("pupil", (count, 2)),
            ("gaze_3d", (count, 3)),
            ("eye_origin", (count, 2, 3)),
            ("eye_direction", (count, 2, 3)),
        ):
            array = getattr(self, name)
            if array is not None and array.shape != expected:
                raise ValueError(
                    f"{name} has shape {array.shape}, expected {expected} "
                    f"for {count} samples"
                )

    def __len__(self) -> int:
        return len(self.time)

    @property
    def valid(self) -> np.ndarray:
        """Which samples the device tracked a gaze point for."""
        return np.isfinite(self.gaze).all(axis=1)

    @property
    def duration(self) -> float:
        """Seconds from the first sample to the last."""
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    @property
    def sample_rate(self) -> float:
        """Samples per second from the median interval; zero with fewer than two samples."""
        return _sample_rate(self.time)

    def gaze_pixels(self, resolution: tuple[int, int] | None = None) -> np.ndarray:
        """Convert gaze to pixels using ``resolution`` or the recording's frame size.

        Coordinates are not clipped to the frame.
        """
        size = resolution or (self.recording.resolution if self.recording else None)
        if size is None:
            raise ValueError(
                "gaze is normalised and this recording has no frame size; "
                "pass the video's resolution"
            )
        return self.gaze * np.asarray(size, dtype=float)

    def between(self, start: float, end: float) -> TrackingData:
        """Select samples in the half-open interval ``[start, end)``."""
        return self[_span(self.time, start, end)]

    def __getitem__(self, index) -> TrackingData:
        """Select samples by slice or mask, preserving recording metadata."""
        return replace(
            self,
            time=self.time[index],
            gaze=self.gaze[index],
            pupil=self.pupil[index],
            gaze_3d=None if self.gaze_3d is None else self.gaze_3d[index],
            eye_origin=None if self.eye_origin is None else self.eye_origin[index],
            eye_direction=(
                None if self.eye_direction is None else self.eye_direction[index]
            ),
        )

    def mean_gaze(self, start: float, end: float) -> np.ndarray:
        """Mean normalised gaze over ``[start, end)``, ignoring NaNs.

        Return NaNs if no finite readings are present.
        """
        window = self.gaze[_span(self.time, start, end)]
        if not np.isfinite(window).any():
            return np.array([np.nan, np.nan])
        return np.nanmean(window, axis=0)


@dataclass(frozen=True)
class Series:
    """Three-axis sensor readings with their own timestamps."""

    #: Seconds on the video container clock, one per reading.
    time: np.ndarray
    #: ``(n, 3)`` readings, one row per time.
    values: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.time), 3)
        if self.values.shape != expected:
            raise ValueError(
                f"values has shape {self.values.shape}, expected {expected}"
            )

    def __len__(self) -> int:
        return len(self.time)

    @property
    def sample_rate(self) -> float:
        """Readings per second from the median interval; zero with fewer than two readings."""
        return _sample_rate(self.time)

    def between(self, start: float, end: float) -> Series:
        """Select readings in the half-open interval ``[start, end)``."""
        index = _span(self.time, start, end)
        return Series(time=self.time[index], values=self.values[index])

    @classmethod
    def empty(cls) -> Series:
        """Create an empty sensor series."""
        return cls(time=np.zeros(0), values=np.zeros((0, 3)))


@dataclass(frozen=True)
class MotionData:
    """Acceleration and angular velocity on the container clock.

    Axes use head coordinates: X left, Y up, Z forward.
    Each sensor has its own timestamps; magnetometer data is omitted.
    """

    #: Acceleration in m/s², gravity included: at rest this is ~9.8 straight up.
    acceleration: Series
    #: Rate of turn in degrees per second about each axis.
    angular_velocity: Series
    #: The source these readings were read from.
    recording: Recording | None = None

    def __len__(self) -> int:
        return max(len(self.acceleration), len(self.angular_velocity))

    def angular_velocity_bias(self, still: float = 0.5) -> np.ndarray:
        """Estimate gyroscope bias from steady one-second sample blocks.

        Average blocks whose per-axis standard deviations are below ``still``
        (degrees/s); return NaNs if none qualify. Constant rotation can also
        qualify as steady.
        """
        rate = self.angular_velocity.sample_rate
        # Round to avoid truncating rates such as 99.99999 Hz to 99 samples.
        window = round(rate) if rate > 0 else 0
        values = self.angular_velocity.values
        if window < 2 or len(values) < window:
            return np.full(3, np.nan)
        blocks = values[: len(values) // window * window].reshape(-1, window, 3)
        steady = blocks.std(axis=1).max(axis=1) < still
        if not steady.any():
            return np.full(3, np.nan)
        return blocks[steady].reshape(-1, 3).mean(axis=0)

    @classmethod
    def empty(cls, recording: Recording | None = None) -> MotionData:
        """Create empty motion data with optional recording metadata."""
        return cls(
            acceleration=Series.empty(),
            angular_velocity=Series.empty(),
            recording=recording,
        )


@dataclass(frozen=True)
class Streams:
    """Gaze and motion returned by one recording read."""

    tracking: TrackingData
    motion: MotionData
