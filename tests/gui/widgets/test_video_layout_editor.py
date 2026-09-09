import pytest

from body_eye_sync.export.layout import LayoutKind
from body_eye_sync.gui.widgets.video_layout_editor import VideoLayoutEditor


@pytest.fixture
def editor(qtbot):
    widget = VideoLayoutEditor()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def _choose(editor, index: int, text: str) -> None:
    """Pick a video in one slot's drop-down, as a user would."""
    editor.boxes[index].combo.setCurrentIndex(editor.boxes[index].combo.findText(text))


def test_the_grid_holds_every_video_in_order(editor):
    editor.set_videos(["room", "side", "glasses"])

    assert editor.layout_kind() is LayoutKind.grid
    assert editor.slots() == ["room", "side", "glasses"]
    assert len(editor.boxes) == 3
    assert editor.unplaced_videos() == []
    assert not editor.unplaced_label.isVisible()


def test_four_plus_one_always_offers_five_slots(editor, qtbot):
    editor.set_videos(["room", "side"])
    changes = []
    editor.changed.connect(lambda: changes.append(editor.slots()))

    editor.set_layout_kind("4+1")

    assert len(editor.boxes) == 5
    assert editor.slots() == ["room", "side", None, None, None]
    assert changes == [["room", "side", None, None, None]]
    assert editor.unplaced_videos() == []


def test_two_plus_one_offers_three_slots(editor):
    editor.set_videos(["room", "side", "glasses", "ceiling"])

    editor.set_layout_kind("2+1")

    assert len(editor.boxes) == 3
    assert editor.slots() == ["room", "side", "glasses"]
    assert editor.unplaced_videos() == ["ceiling"]


def test_a_layout_that_cannot_show_every_video_says_so(editor):
    editor.set_layout_kind("4+1")

    editor.set_videos([f"camera{index}" for index in range(6)])

    assert editor.slots() == ["camera0", "camera1", "camera2", "camera3", "camera4"]
    assert editor.unplaced_videos() == ["camera5"]
    assert editor.unplaced_label.isVisible()
    assert "camera5" in editor.unplaced_label.text()


def test_choosing_a_placed_video_swaps_the_two_slots(editor):
    editor.set_videos(["room", "side", "glasses"])
    changes = []
    editor.changed.connect(lambda: changes.append(editor.slots()))

    _choose(editor, 0, "glasses")

    assert editor.slots() == ["glasses", "side", "room"]
    assert changes == [["glasses", "side", "room"]]
    assert [box.combo.currentText() for box in editor.boxes] == [
        "glasses",
        "side",
        "room",
    ]


def test_a_slot_can_be_emptied_and_filled_again(editor):
    editor.set_videos(["room", "side"])
    editor.set_layout_kind("4+1")

    _choose(editor, 0, "(empty)")

    assert editor.slots() == [None, "side", None, None, None]
    assert editor.unplaced_videos() == ["room"]

    _choose(editor, 3, "room")

    assert editor.slots() == [None, "side", None, "room", None]
    assert editor.unplaced_videos() == []


def test_deselected_videos_leave_the_other_placements_alone(editor):
    editor.set_videos(["room", "side", "glasses"])
    editor.set_layout_kind("4+1")
    _choose(editor, 4, "glasses")

    editor.set_videos(["room", "glasses"])

    assert editor.slots() == ["room", None, None, None, "glasses"]


def test_the_boxes_sit_where_their_videos_will_be_drawn(editor, qtbot):
    editor.set_videos(["centre", "corner"])
    editor.set_layout_kind("4+1")
    qtbot.waitUntil(lambda: editor.boxes[0].width() > 0)

    frame = editor.canvas.frame_rect()
    centre, top_left = editor.boxes[0].geometry(), editor.boxes[1].geometry()

    assert frame.contains(centre) and frame.contains(top_left)
    assert top_left.topLeft() == frame.topLeft()
    assert centre.center().x() == pytest.approx(frame.center().x(), abs=1)
    assert centre.center().y() == pytest.approx(frame.center().y(), abs=1)
    # The corner covers a part of the centre, and is drawn on top of it.
    assert top_left.intersects(centre)
    assert editor.canvas.children().index(
        editor.boxes[1]
    ) > editor.canvas.children().index(editor.boxes[0])
