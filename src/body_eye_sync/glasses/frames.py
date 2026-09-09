"""Convert IMU readings to head coordinates: X left, Y up, Z forward."""

from __future__ import annotations

import numpy as np

#: Glasses 3 IMU mounting angle in degrees.
MOUNT_ANGLE = 12.0

_COS = np.cos(np.radians(MOUNT_ANGLE))
_SIN = np.sin(np.radians(MOUNT_ANGLE))

#: IMU-to-head transform for Glasses 3 firmware before 1.29.
GLASSES3_IMU = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -_COS, _SIN],
        [0.0, _SIN, _COS],
    ]
)

#: Glasses 2 Y-axis correction; no mounting-angle correction is applied.
GLASSES2_IMU = np.diag([1.0, -1.0, 1.0])


def to_head_unit(values: np.ndarray, rotation: np.ndarray | None) -> np.ndarray:
    """``values`` as ``(n, 3)`` rows in the head unit coordinate system."""
    values = np.asarray(values, dtype=float).reshape(-1, 3)
    return values if rotation is None else values @ rotation.T
