"""Where an input's own clock sits on the shared experiment timeline.

Every recording is made on its own device and starts whenever that device was
switched on, so each carries a ``time_offset`` saying how far its clock is from
the experiment's. It may also have a ``time_scale`` for a clock that runs slightly fast or
slow, and if it loses content it may have a list of
:class:`~body_eye_sync.preprocessing.alignment.Shift`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from body_eye_sync.experiment.config import TimeShift
from body_eye_sync.preprocessing.alignment import (
    Shift,
    to_local_time,
    to_experiment_time,
    unobserved,
)


class Timeline:
    """An input's conversions between its own local clock and the experiment's."""

    id: str

    def __init__(
        self,
        time_offset: float = 0.0,
        time_scale: float = 1.0,
        time_shifts: Iterable[Shift] = (),
    ) -> None:
        self.time_offset = time_offset
        self.time_scale = time_scale
        self.time_shifts: list[Shift] = list(time_shifts)

    @property
    def path(self) -> Path | None:
        """The recording this input reads, whatever kind of media it is."""
        raise NotImplementedError

    @staticmethod
    def timeline_kwargs(spec) -> dict:
        """The timeline an input's stored form records, as constructor arguments."""
        return {
            "time_offset": spec.time_offset,
            "time_scale": spec.time_scale,
            "time_shifts": [Shift(s.at, s.seconds) for s in spec.time_shifts],
        }

    def timeline_config(self) -> dict:
        """The timeline fields in the form an input's config model takes."""
        return {
            "time_offset": self.time_offset,
            "time_scale": self.time_scale,
            "time_shifts": [
                TimeShift(at=shift.at, seconds=shift.seconds)
                for shift in self.time_shifts
            ],
        }

    def to_experiment_time(self, local_time: float) -> float:
        """Experiment time for a moment on this input's own clock."""
        return to_experiment_time(
            local_time, self.time_offset, self.time_shifts, self.time_scale
        )

    def to_local_time(self, experiment_time: float) -> float | None:
        """This input's own clock at a moment of the experiment.

        ``None`` if the input has nothing for that moment, either because it
        lost the content or because it was not recording yet.
        """
        return to_local_time(
            experiment_time, self.time_offset, self.time_shifts, self.time_scale
        )

    def unobserved(self) -> list[tuple[float, float]]:
        """Stretches of experiment time this input has no recording of."""
        return unobserved(self.time_offset, self.time_shifts, self.time_scale)
