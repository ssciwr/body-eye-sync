from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.video_viewer import VideoViewer


def _video(data_dir, tracklets=None):
    video = Video()
    video.set_video(data_dir / "three-people.mp4")
    if tracklets is not None:
        video.set_data(tracklets)
    return video


def _one_box_on_first_frame():
    # A box on the first frame (index 0).
    return pd.DataFrame(
        {
            "frame": [0],
            "track_id": [1],
            "x1": [0.0],
            "y1": [0.0],
            "x2": [10.0],
            "y2": [10.0],
            "conf": [0.9],
        }
    )


@pytest.fixture
def viewer(qtbot, data_dir):
    widget = VideoViewer()
    qtbot.addWidget(widget)
    widget.load(_video(data_dir))
    return widget


def test_video_viewer_load_shows_first_frame(viewer):
    assert viewer.frame_count == 5
    assert viewer.current_frame == 0
    assert not viewer._pixmap_item.pixmap().isNull()


def test_video_viewer_load_clears_stale_overlays(qtbot, data_dir):
    widget = VideoViewer()
    qtbot.addWidget(widget)

    # A video whose first frame has a box draws it on load.
    widget.load(_video(data_dir, _one_box_on_first_frame()))
    assert len(widget._overlay_items) == 2

    # Loading a video with no object tracking results must drop those overlays.
    widget.load(_video(data_dir))
    assert widget._overlay_items == []


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


def test_video_viewer_corrects_overestimated_frame_count(viewer):
    # CAP_PROP_FRAME_COUNT often over-counts; pretend it claimed 8 frames when
    # the test video really has 5 (indices 0-4).
    viewer._set_frame_count(8)

    # Seeking to the bogus end steps back to the real last frame and shrinks the
    # count to the number of frames that actually decode.
    viewer.set_frame(7)
    assert viewer.current_frame == 4
    assert viewer.frame_count == 5


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


def test_video_viewer_set_transport_enabled_toggles_controls(viewer):
    viewer.enable_controls(False)
    assert not viewer._play_button.isEnabled()
    assert not viewer._slider.isEnabled()
    assert not viewer._spinbox.isEnabled()

    viewer.enable_controls(True)
    assert viewer._play_button.isEnabled()
    assert viewer._slider.isEnabled()
    assert viewer._spinbox.isEnabled()


def test_video_viewer_disabling_transport_stops_playback(viewer):
    viewer._play_button.setChecked(True)
    assert viewer._timer.isActive()

    viewer.enable_controls(False)
    assert not viewer._timer.isActive()
    assert not viewer._play_button.isChecked()


def test_video_viewer_show_live_frame_draws_its_own_boxes(viewer):
    # The video has no tracklets, so its boxes come straight from the frame's
    # tracks (BoxMOT layout: x1, y1, x2, y2, id, conf, cls, det_ind).
    frame = SimpleNamespace(
        frame_idx=2, tracks=np.array([[0.0, 0.0, 10.0, 10.0, 1, 0.9, 0, 0]])
    )
    viewer.show_live_frame(frame)

    # Jumped to the tracked frame (1-based 2 -> 0-based 1) and drew its one box.
    assert viewer.current_frame == 1
    assert len(viewer._overlay_items) == 2  # one rect + one label


def test_video_viewer_draws_boxes_from_video(qtbot, data_dir):
    # One box on frame 0 (tracker frame 1), none elsewhere.
    viewer = VideoViewer()
    qtbot.addWidget(viewer)
    viewer.load(_video(data_dir, _one_box_on_first_frame()))

    # One box -> one rect + one label.
    assert len(viewer._overlay_items) == 2

    # Overlays update themselves when the frame changes.
    viewer.set_frame(1)
    assert viewer._overlay_items == []

    viewer.set_frame(0)
    assert len(viewer._overlay_items) == 2
