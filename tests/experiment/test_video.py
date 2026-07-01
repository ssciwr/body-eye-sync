from types import SimpleNamespace

import numpy as np
import pandas as pd

from body_eye_sync.experiment.video import Video


def _data():
    return pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "track_id": [1, 2, 1],
            "x1": [0.0, 10.0, 1.0],
            "y1": [0.0, 10.0, 1.0],
            "x2": [5.0, 15.0, 6.0],
            "y2": [5.0, 15.0, 6.0],
            "conf": [0.9, 0.8, 0.95],
        }
    )


def _frame(frame_idx, *boxes):
    """A stand-in for a BoxMOT frame result: ``frame_idx`` plus a tracks array.

    Each box is ``(x1, y1, x2, y2, track_id, conf)``; rows follow BoxMOT's
    layout ``x1, y1, x2, y2, id, conf, cls, det_ind``.
    """
    rows = [[x1, y1, x2, y2, tid, conf, 0, 0] for x1, y1, x2, y2, tid, conf in boxes]
    tracks = np.array(rows) if rows else np.empty((0, 8))
    return SimpleNamespace(frame_idx=frame_idx, tracks=tracks)


def test_boxes_for_frame_empty_without_tracklets():
    assert Video().boxes_for_frame(0) == []


def test_boxes_for_frame_looks_up_by_index():
    video = Video()
    video.set_data(_data())

    boxes = video.boxes_for_frame(0)
    assert {b.object_id for b in boxes} == {1, 2}

    boxes = video.boxes_for_frame(1)
    assert [b.object_id for b in boxes] == [1]


def test_no_boxes_exposed_while_tracking():
    video = Video()
    video.add_frame(_frame(1, (0.0, 0.0, 5.0, 5.0, 7, 0.9)))

    # Frames are drawn live straight from the worker, so nothing is exposed here
    # and the DataFrame stays unavailable until the run is marked complete.
    assert video.boxes_for_frame(0) == []
    assert video.data is None


def test_finish_tracking_collapses_streamed_frames():
    video = Video()
    video.add_frame(_frame(1, (0.0, 0.0, 5.0, 5.0, 1, 0.9)))
    video.add_frame(_frame(2, (1.0, 1.0, 6.0, 6.0, 1, 0.8)))

    # BoxMOT frames 1 and 2 are stored 0-based as 0 and 1.
    video.finish_tracking()
    assert [b.x1 for b in video.boxes_for_frame(0)] == [0.0]
    assert [b.x1 for b in video.boxes_for_frame(1)] == [1.0]


def test_begin_tracking_clears_previous_results():
    video = Video()
    video.set_data(_data())
    video.begin_tracking()
    assert video.data is None
    assert video.boxes_for_frame(0) == []


def test_set_video_invalidates_tracklets():
    video = Video()
    video.set_data(_data())
    video.set_video("some/video.mp4")
    assert video.data is None
    assert video.boxes_for_frame(0) == []
