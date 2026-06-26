import pytest

from body_eye_sync.state import BoundingBox
from body_eye_sync.video_viewer import VideoViewer


@pytest.fixture
def viewer(qtbot, data_dir):
    widget = VideoViewer()
    qtbot.addWidget(widget)
    widget.load(data_dir / "three-people.mp4")
    return widget


def test_video_viewer_load_shows_first_frame(viewer):
    assert viewer.frame_count == 5
    assert viewer.current_frame == 0
    assert not viewer._pixmap_item.pixmap().isNull()


def test_video_viewer_seek_forward_and_back(viewer):
    viewer.set_frame(3)
    assert viewer.current_frame == 3
    # Seeking backwards exercises the re-seek path.
    viewer.set_frame(1)
    assert viewer.current_frame == 1


def test_video_viewer_set_frame_clamps_to_valid_range(viewer):
    viewer.set_frame(999)
    assert viewer.current_frame == viewer.frame_count - 1
    viewer.set_frame(-5)
    assert viewer.current_frame == 0


def test_video_viewer_advance_steps_one_frame(viewer):
    viewer.set_frame(0)
    viewer._advance()
    assert viewer.current_frame == 1


def test_video_viewer_frame_changed_signal_emits_index(viewer):
    seen = []
    viewer.frame_changed.connect(seen.append)
    viewer.set_frame(2)
    viewer.set_frame(4)
    assert seen == [2, 4]


def test_video_viewer_draws_boxes_from_provider(viewer):
    # One box on frame 0, none elsewhere.
    def provider(frame_index):
        if frame_index == 0:
            return [BoundingBox(0, 0, 10, 10, track_id=1, conf=0.9)]
        return []

    viewer.set_frame(0)
    viewer.set_box_provider(provider)
    # One box -> one rect + one label.
    assert len(viewer._overlay_items) == 2

    # Overlays update themselves when the frame changes.
    viewer.set_frame(1)
    assert viewer._overlay_items == []
