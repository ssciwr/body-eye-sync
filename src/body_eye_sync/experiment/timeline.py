"""Where an input's own clock sits on the shared experiment timeline.

Every recording is made on its own device and starts whenever that device was
switched on, so each owns a :class:`Timeline` with the offset of its clock from
the experiment's and any gaps in content it lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from body_eye_sync.experiment.config import TimelineConfig, TimeShiftConfig


@dataclass
class Shift:
    """A stretch of a recording that was never written."""

    at: float
    seconds: float


@dataclass
class Timeline:
    """Conversions between one recording's clock and the experiment's."""

    offset: float = 0.0
    shifts: list[Shift] = field(default_factory=list)

    @property
    def corrects_timing(self) -> bool:
        return bool(self.shifts)

    @classmethod
    def from_config(cls, config: TimelineConfig) -> Self:
        """Build runtime timeline state from its serialisable form."""
        return cls(
            offset=config.offset,
            shifts=[Shift(shift.at, shift.seconds) for shift in config.shifts],
        )

    def to_config(self) -> TimelineConfig:
        """Return the serialisable form of this runtime timeline."""
        return TimelineConfig(
            offset=self.offset,
            shifts=[
                TimeShiftConfig(at=shift.at, seconds=shift.seconds)
                for shift in self.shifts
            ],
        )

    def to_experiment_time(self, local_time: float) -> float:
        """Experiment time for a moment on this input's own clock."""
        return to_experiment_time(local_time, self.offset, self.shifts)

    def to_local_time(self, experiment_time: float) -> float | None:
        """This input's own clock at a moment of the experiment.

        ``None`` if the input has nothing for that moment, either because it
        lost the content or because it was not recording yet.
        """
        return to_local_time(experiment_time, self.offset, self.shifts)

    def unobserved(self) -> list[tuple[float, float]]:
        """Stretches of experiment time this input has no recording of."""
        return unobserved(self.offset, self.shifts)


def missing_before(shifts: list[Shift], local_time: float) -> float:
    """How much content is missing before local_time on this recording's clock."""
    return sum(shift.seconds for shift in shifts if shift.at <= local_time)


def to_experiment_time(
    local_time: float,
    offset: float,
    shifts: list[Shift],
) -> float:
    """Experiment time for a moment on one recording's own clock.

    ``offset`` is where the recording starts, and each :class:`Shift` before
    ``local_time`` adds the content missing at that point. Always defined:
    every moment the recording holds did happen.
    """
    return local_time + offset + missing_before(shifts, local_time)


def to_local_time(
    experiment_time: float,
    offset: float,
    shifts: list[Shift],
) -> float | None:
    """Where a moment of the experiment sits on one recording's own clock.

    ``None`` when the recording holds no such moment, which is what a
    :class:`Shift` means: the content was never written, so a stretch of the
    experiment has nowhere to map to.
    """
    running = offset
    for shift in sorted(shifts, key=lambda value: value.at):
        if experiment_time < shift.at + running:
            return experiment_time - running
        running += shift.seconds
        if experiment_time < shift.at + running:
            return None
    return experiment_time - running


def unobserved(offset: float, shifts: list[Shift]) -> list[tuple[float, float]]:
    """The stretches of experiment time a recording has nothing for.

    One per :class:`Shift`, as ``(start, end)`` on the experiment clock.
    """
    spans: list[tuple[float, float]] = []
    running = offset
    for shift in sorted(shifts, key=lambda value: value.at):
        start = shift.at + running
        running += shift.seconds
        spans.append((start, shift.at + running))
    return spans
