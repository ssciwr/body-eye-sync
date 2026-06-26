import pandas as pd

from body_eye_sync.state import AppState


def _tracklets():
    return pd.DataFrame(
        {
            "frame": [1, 1, 2],
            "track_id": [1, 2, 1],
            "x1": [0.0, 10.0, 1.0],
            "y1": [0.0, 10.0, 1.0],
            "x2": [5.0, 15.0, 6.0],
            "y2": [5.0, 15.0, 6.0],
            "conf": [0.9, 0.8, 0.95],
        }
    )


def test_boxes_for_frame_empty_without_tracklets():
    assert AppState().boxes_for_frame(0) == []


def test_boxes_for_frame_maps_zero_based_to_one_based():
    state = AppState()
    state.set_tracklets(_tracklets())

    # Viewer frame 0 corresponds to tracker frame 1.
    boxes = state.boxes_for_frame(0)
    assert {b.track_id for b in boxes} == {1, 2}

    boxes = state.boxes_for_frame(1)
    assert [b.track_id for b in boxes] == [1]
    assert boxes[0].conf == 0.95


def test_set_video_invalidates_tracklets():
    state = AppState()
    state.set_tracklets(_tracklets())
    state.set_video("some/video.mp4")
    assert state.tracklets is None
    assert state.boxes_for_frame(0) == []
