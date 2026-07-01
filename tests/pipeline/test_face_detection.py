import numpy as np

from body_eye_sync.pipeline.object_tracking import BoundingBox
from body_eye_sync.pipeline.face_detection import (
    FACE_COLUMNS,
    LANDMARK_NAMES,
    FaceBox,
    FaceFrameResult,
    face_box_from_row,
    faces_to_dataframe,
)


def _face(track_id, score=0.9):
    # five landmarks, just offset from the box origin so they're distinct
    landmarks = [(10.0 + i, 20.0 + i) for i in range(len(LANDMARK_NAMES))]
    box = BoundingBox(10.0, 20.0, 30.0, 50.0, track_id)
    return FaceBox(box, score, landmarks)


def test_faces_to_dataframe_columns_and_dtypes():
    df = faces_to_dataframe([FaceFrameResult(0, [_face(1), _face(2)])])

    assert list(df.columns) == ["frame", "track_id", *FACE_COLUMNS]
    assert df["frame"].dtype == int
    assert df["track_id"].dtype == int
    assert df["face_score"].dtype == float
    assert df["track_id"].tolist() == [1, 2]


def test_faces_to_dataframe_skips_frames_without_faces():
    df = faces_to_dataframe([FaceFrameResult(0, []), FaceFrameResult(1, [_face(3)])])

    assert df["frame"].tolist() == [1]
    assert df["track_id"].tolist() == [3]


def test_faces_to_dataframe_empty():
    df = faces_to_dataframe([])

    assert list(df.columns) == ["frame", "track_id", *FACE_COLUMNS]
    assert len(df) == 0


def test_face_box_from_row_round_trips():
    df = faces_to_dataframe([FaceFrameResult(2, [_face(4, score=0.7)])])
    row = next(df.itertuples(index=False))

    face = face_box_from_row(row)
    assert face.box.track_id == 4
    assert face.score == 0.7
    assert (face.box.x1, face.box.y1, face.box.x2, face.box.y2) == (
        10.0,
        20.0,
        30.0,
        50.0,
    )
    assert face.landmarks == [(10.0 + i, 20.0 + i) for i in range(len(LANDMARK_NAMES))]


def test_landmark_columns_are_two_per_name():
    # bbox(4) + score(1) + two coords per landmark
    assert FACE_COLUMNS == [
        "face_score",
        "face_x1",
        "face_y1",
        "face_x2",
        "face_y2",
        *[f"{n}_{a}" for n in LANDMARK_NAMES for a in ("x", "y")],
    ]
    assert not np.isnan(_face(1).score)
