"""Where an input's own clock sits on the shared experiment timeline.

Every recording is made on its own device, which starts whenever that device
was switched on and counts time on its own crystal. Two crystals differ by tens
of parts per million, so a :class:`Timeline` contains an offset and a rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from body_eye_sync.experiment.config import TimelineConfig


@dataclass
class Timeline:
    """Conversions between one recording's clock and the experiment's."""

    offset: float = 0.0
    #: Experiment seconds per second of this recording's own clock.
    rate: float = 1.0

    @property
    def drift_ppm(self) -> float:
        """How far this recording's clock runs from the experiment's, in ppm."""
        return (self.rate - 1.0) * 1e6

    @property
    def corrects_drift(self) -> bool:
        """Whether this timeline carries a measured clock-rate difference."""
        return self.rate != 1.0

    @classmethod
    def from_config(cls, config: TimelineConfig) -> Self:
        """Build runtime timeline state from its serialisable form."""
        return cls(offset=config.offset, rate=config.rate)

    def to_config(self) -> TimelineConfig:
        """Return the serialisable form of this runtime timeline."""
        return TimelineConfig(offset=self.offset, rate=self.rate)

    def to_experiment_time(self, local_time: float) -> float:
        """Experiment time for a moment on this input's own clock."""
        return to_experiment_time(local_time, self.offset, self.rate)

    def to_experiment_times(self, local_times: np.ndarray) -> np.ndarray:
        """Experiment times for an array of moments on this input's clock."""
        return self.offset + np.asarray(local_times, dtype=float) * self.rate

    def to_local_time(self, experiment_time: float) -> float:
        """This input's own clock at a moment of the experiment."""
        return to_local_time(experiment_time, self.offset, self.rate)


def to_experiment_time(local_time: float, offset: float, rate: float = 1.0) -> float:
    """Experiment time for a moment on one recording's own clock.

    ``offset`` is where the recording starts and ``rate`` is how fast its clock
    runs against the experiment's. Always defined: every moment the recording's
    clock names did happen, whether or not content was captured for it.
    """
    return offset + local_time * rate


def to_local_time(experiment_time: float, offset: float, rate: float = 1.0) -> float:
    """Where a moment of the experiment sits on one recording's own clock.

    Outside the recording's own duration the result is simply out of range;
    callers that stream a recording check that themselves.
    """
    return (experiment_time - offset) / rate
