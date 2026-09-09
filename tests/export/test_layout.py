import pytest

from body_eye_sync.export.layout import (
    LayoutKind,
    Placement,
    layout_frame,
    slot_count,
)


def test_the_grid_grows_to_hold_every_video_and_the_other_layouts_do_not():
    assert slot_count(LayoutKind.grid, 3) == 3
    assert slot_count("grid", 7) == 7
    # An empty selection still leaves one slot to fill.
    assert slot_count(LayoutKind.grid, 0) == 1
    assert slot_count(LayoutKind.two_plus_one, 1) == 3
    assert slot_count("2+1", 9) == 3
    assert slot_count(LayoutKind.four_plus_one, 2) == 5
    assert slot_count("4+1", 9) == 5


def test_a_grid_fills_square_rows_from_the_top_left():
    frame = layout_frame(LayoutKind.grid, 3, (64, 48))

    # Three videos in two columns, so the second row is half empty.
    assert (frame.width, frame.height) == (128, 96)
    assert frame.placements == (
        Placement(0, 0, 64, 48),
        Placement(64, 0, 64, 48),
        Placement(0, 48, 64, 48),
    )


def test_two_plus_one_centres_its_third_video_under_the_other_two():
    frame = layout_frame(LayoutKind.two_plus_one, 3, (64, 48))

    assert (frame.width, frame.height) == (128, 96)
    assert frame.placements == (
        Placement(0, 0, 64, 48),
        Placement(64, 0, 64, 48),
        Placement(32, 48, 64, 48),
    )


def test_the_four_plus_one_corners_overlap_its_central_video():
    frame = layout_frame(LayoutKind.four_plus_one, 5, (64, 48))

    assert (frame.width, frame.height) == (192, 144)
    centre, *corners = frame.placements
    assert centre == Placement(62, 47, 67, 50)
    # One corner of the frame each, drawn after -- and so over -- the centre.
    assert [(corner.x, corner.y) for corner in corners] == [
        (0, 0),
        (118, 0),
        (0, 89),
        (118, 89),
    ]
    assert all((corner.width, corner.height) == (74, 55) for corner in corners)
    for corner in corners:
        assert corner.x < centre.x + centre.width
        assert centre.x < corner.x + corner.width
        assert corner.y < centre.y + centre.height
        assert centre.y < corner.y + corner.height


def test_a_layout_needs_slots_a_shape_and_a_known_kind():
    with pytest.raises(ValueError, match="at least one slot"):
        layout_frame(LayoutKind.grid, 0)
    with pytest.raises(ValueError, match="cell dimensions must be positive"):
        layout_frame(LayoutKind.grid, 1, (0, 48))
    with pytest.raises(ValueError):
        layout_frame("mosaic", 1)
