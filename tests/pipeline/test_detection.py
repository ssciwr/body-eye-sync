import numpy as np

from body_eye_sync.pipeline.detection import (
    BoundingBox,
    boxes_from_tracks,
    tracks_to_dataframe,
)


def _tracks_frame1():
    # BoxMOT tracks layout: x1, y1, x2, y2, id, conf, cls, det_ind
    return np.array(
        [
            [0.0, 0.0, 5.0, 5.0, 1, 0.9, 0, 0],
            [10.0, 10.0, 15.0, 15.0, 2, 0.8, 0, 1],
        ]
    )


def _tracks_frame2():
    return np.array([[1.0, 1.0, 6.0, 6.0, 1, 0.95, 0, 0]])


def test_boxes_from_tracks():
    boxes = boxes_from_tracks(_tracks_frame1())

    assert boxes[0] == BoundingBox(0.0, 0.0, 5.0, 5.0, object_id=1)
    assert {b.object_id for b in boxes} == {1, 2}


def test_boxes_from_tracks_empty():
    assert boxes_from_tracks(np.empty((0, 8))) == []


def test_tracks_to_dataframe_is_numeric():
    df = tracks_to_dataframe([(0, _tracks_frame1()), (1, _tracks_frame2())])

    assert list(df.columns) == ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
    assert df["frame"].dtype == int
    assert df["track_id"].dtype == int
    assert df["conf"].dtype == float
    assert df["frame"].tolist() == [0, 0, 1]
    assert df["track_id"].tolist() == [1, 2, 1]


def test_tracks_to_dataframe_skips_empty_frames():
    df = tracks_to_dataframe([(0, np.empty((0, 8))), (1, _tracks_frame2())])

    assert df["frame"].tolist() == [1]
    assert df["track_id"].tolist() == [1]
