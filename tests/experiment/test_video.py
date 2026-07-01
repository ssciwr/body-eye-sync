from types import SimpleNamespace

import numpy as np
import pandas as pd

from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.object_tracking import BoundingBox
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult


def _face_result(frame_idx, *faces):
    """A FaceFrameResult; each face is ``(track_id, x1, y1, x2, y2, score)``."""
    boxes = [
        FaceBox(BoundingBox(x1, y1, x2, y2, tid), score, [(x1, y1)] * 5)
        for tid, x1, y1, x2, y2, score in faces
    ]
    return FaceFrameResult(frame_idx, boxes)


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
    assert {b.track_id for b in boxes} == {1, 2}

    boxes = video.boxes_for_frame(1)
    assert [b.track_id for b in boxes] == [1]


def test_no_boxes_exposed_while_object_tracking():
    video = Video()
    video.add_object_tracking_frame(_frame(1, (0.0, 0.0, 5.0, 5.0, 7, 0.9)))

    # Frames are drawn live straight from the worker, so nothing is exposed here
    # and the DataFrame stays unavailable until the run is marked complete.
    assert video.boxes_for_frame(0) == []
    assert video.data is None


def test_finish_object_tracking_collapses_streamed_frames():
    video = Video()
    video.add_object_tracking_frame(_frame(1, (0.0, 0.0, 5.0, 5.0, 1, 0.9)))
    video.add_object_tracking_frame(_frame(2, (1.0, 1.0, 6.0, 6.0, 1, 0.8)))

    # BoxMOT frames 1 and 2 are stored 0-based as 0 and 1.
    video.finish_object_tracking()
    assert [b.x1 for b in video.boxes_for_frame(0)] == [0.0]
    assert [b.x1 for b in video.boxes_for_frame(1)] == [1.0]


def test_begin_object_tracking_clears_previous_results():
    video = Video()
    video.set_data(_data())
    video.begin_object_tracking()
    assert video.data is None
    assert video.boxes_for_frame(0) == []


def test_discard_object_tracking_drops_partial_output():
    video = Video()
    video.add_object_tracking_frame(_frame(1, (0.0, 0.0, 5.0, 5.0, 1, 0.9)))
    video.discard_object_tracking()

    # A cancelled/failed run leaves nothing behind.
    assert video.data is None
    assert video.boxes_for_frame(0) == []


def test_discard_face_detection_keeps_tracked_boxes():
    video = Video()
    video.set_data(_data())
    video.begin_face_detection()
    video.add_face_detection_frame(_face_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.9)))
    video.discard_face_detection()

    # The tracked boxes survive; the aborted face pass merges nothing.
    assert len(video.data) == 3
    assert "face_score" not in video.data.columns
    assert video.faces_for_frame(0) == []


def test_set_video_invalidates_tracklets():
    video = Video()
    video.set_data(_data())
    video.set_video("some/video.mp4")
    assert video.data is None
    assert video.boxes_for_frame(0) == []


def test_all_boxes_by_frame_groups_tracked_boxes():
    video = Video()
    video.set_data(_data())

    boxes = video.all_boxes_by_frame()
    assert set(boxes) == {0, 1}
    assert {b.track_id for b in boxes[0]} == {1, 2}  # track ids in frame 0
    assert boxes[1] == [BoundingBox(1.0, 1.0, 6.0, 6.0, 1)]


def test_finish_face_detection_merges_onto_track_rows():
    video = Video()
    video.set_data(_data())

    # A face for (frame 0, track 1) and (frame 1, track 1); (frame 0, track 2) none.
    video.add_face_detection_frame(_face_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.9)))
    video.add_face_detection_frame(_face_result(1, (1, 1.0, 1.0, 5.0, 5.0, 0.8)))
    video.finish_face_detection()

    data = video.data
    # Original rows preserved, face columns added, the missing face left as NaN.
    assert len(data) == 3
    assert "face_score" in data.columns
    assert data["face_score"].notna().sum() == 2

    faces0 = video.faces_for_frame(0)
    assert [f.box.track_id for f in faces0] == [1]
    assert faces0[0].score == 0.9
    assert len(faces0[0].landmarks) == 5

    # frame 1 track 1 has a face; (frame 0, track 2) has none.
    assert [f.box.track_id for f in video.faces_for_frame(1)] == [1]


def test_faces_for_frame_empty_without_detection():
    video = Video()
    video.set_data(_data())
    assert video.faces_for_frame(0) == []


def test_begin_face_detection_drops_previous_face_columns():
    video = Video()
    video.set_data(_data())
    video.add_face_detection_frame(_face_result(0, (1, 0.0, 0.0, 4.0, 4.0, 0.9)))
    video.finish_face_detection()
    assert "face_score" in video.data.columns

    video.begin_face_detection()
    # Tracks survive, the stale face columns do not.
    assert "face_score" not in video.data.columns
    assert len(video.data) == 3
    assert video.faces_for_frame(0) == []
