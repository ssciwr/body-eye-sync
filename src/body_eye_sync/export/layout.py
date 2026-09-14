"""Where each video sits in the frame of a combined video."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_CELL_SIZE = (640, 360)


class LayoutKind(StrEnum):
    """The available arrangements of the videos in a combined video."""

    grid = "grid"
    two_plus_one = "2+1"
    four_plus_one = "4+1"


@dataclass(frozen=True)
class Placement:
    """One slot's rectangle in the combined video frame, in pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LayoutFrame:
    """A combined video frame, and the rectangle of each of its slots.

    The slots are in drawing order: a later one is drawn over an earlier one.
    """

    width: int
    height: int
    placements: tuple[Placement, ...]


def slot_count(kind: LayoutKind | str, videos: int) -> int:
    """How many videos this layout shows, given how many are available.

    The grid grows to hold every video; the other layouts have a fixed number
    of slots, which may be left empty.
    """
    match LayoutKind(kind):
        case LayoutKind.grid:
            return max(1, videos)
        case LayoutKind.two_plus_one:
            return 3
        case LayoutKind.four_plus_one:
            return 5


def layout_frame(
    kind: LayoutKind | str,
    slots: int,
    cell_size: tuple[int, int] = DEFAULT_CELL_SIZE,
) -> LayoutFrame:
    """The output frame and slot rectangles of one layout.

    ``cell_size`` is the size of one grid cell, and sets the scale of the whole
    frame.
    """
    if slots <= 0:
        raise ValueError("a layout needs at least one slot")
    if len(cell_size) != 2 or any(value <= 0 for value in cell_size):
        raise ValueError("cell dimensions must be positive")
    match LayoutKind(kind):
        case LayoutKind.two_plus_one:
            return _two_plus_one_frame(cell_size)
        case LayoutKind.four_plus_one:
            return _four_plus_one_frame(cell_size)
        case _:
            return _grid_frame(slots, cell_size)


def _grid_frame(slots: int, cell_size: tuple[int, int]) -> LayoutFrame:
    """Equally sized cells, filling rows from the top left."""
    width, height = cell_size
    column_count = math.ceil(math.sqrt(slots))
    row_count = math.ceil(slots / column_count)
    return LayoutFrame(
        column_count * width,
        row_count * height,
        tuple(
            Placement(
                index % column_count * width, index // column_count * height, *cell_size
            )
            for index in range(slots)
        ),
    )


def _two_plus_one_frame(cell_size: tuple[int, int]) -> LayoutFrame:
    """Two videos side by side, with a third centred in the row below them."""
    width, height = cell_size
    return LayoutFrame(
        2 * width,
        2 * height,
        (
            Placement(0, 0, width, height),
            Placement(width, 0, width, height),
            Placement(width // 2, height, width, height),
        ),
    )


def _four_plus_one_frame(cell_size: tuple[int, int]) -> LayoutFrame:
    """One central video, with a video lapping over each of its corners."""
    cells = 3
    centre_fraction = 0.35
    corner_fraction = 0.385
    width, height = cell_size
    frame_width = cells * width
    frame_height = cells * height
    centre_width = round(centre_fraction * frame_width)
    centre_height = round(centre_fraction * frame_height)
    corner_width = round(corner_fraction * frame_width)
    corner_height = round(corner_fraction * frame_height)
    right = frame_width - corner_width
    bottom = frame_height - corner_height
    return LayoutFrame(
        frame_width,
        frame_height,
        (
            Placement(
                (frame_width - centre_width) // 2,
                (frame_height - centre_height) // 2,
                centre_width,
                centre_height,
            ),
            Placement(0, 0, corner_width, corner_height),
            Placement(right, 0, corner_width, corner_height),
            Placement(0, bottom, corner_width, corner_height),
            Placement(right, bottom, corner_width, corner_height),
        ),
    )
