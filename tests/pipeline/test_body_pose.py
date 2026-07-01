from types import SimpleNamespace

import numpy as np

from body_eye_sync.pipeline.object_tracking import BoundingBox
from body_eye_sync.pipeline.body_pose import (
    KEYPOINT_NAMES,
    POSE_COLUMNS,
    BodyPose,
    DEFAULT_MODEL_NAME,
    PoseFrameResult,
    _detect_in_boxes,
    _resolve_model,
    detect_body_poses,
    default_model_path,
    pose_from_row,
    poses_to_dataframe,
)


def _pose(track_id, score=0.9):
    # COCO keypoints, offset from the box origin so they are distinct.
    keypoints = [
        (10.0 + i, 20.0 + i, 0.5 + i / 100.0) for i in range(len(KEYPOINT_NAMES))
    ]
    box = BoundingBox(10.0, 20.0, 30.0, 50.0, track_id)
    return BodyPose(box, score, keypoints)


def test_poses_to_dataframe_columns_and_dtypes():
    df = poses_to_dataframe([PoseFrameResult(0, [_pose(1), _pose(2)])])

    assert list(df.columns) == ["frame", "track_id", *POSE_COLUMNS]
    assert df["frame"].dtype == int
    assert df["track_id"].dtype == int
    assert df["pose_score"].dtype == float
    assert df["track_id"].tolist() == [1, 2]


def test_poses_to_dataframe_skips_frames_without_poses():
    df = poses_to_dataframe([PoseFrameResult(0, []), PoseFrameResult(1, [_pose(3)])])

    assert df["frame"].tolist() == [1]
    assert df["track_id"].tolist() == [3]


def test_poses_to_dataframe_empty():
    df = poses_to_dataframe([])

    assert list(df.columns) == ["frame", "track_id", *POSE_COLUMNS]
    assert len(df) == 0


def test_pose_from_row_round_trips():
    df = poses_to_dataframe([PoseFrameResult(2, [_pose(4, score=0.7)])])
    row = next(df.itertuples(index=False))

    pose = pose_from_row(row)
    assert pose.box.track_id == 4
    assert pose.score == 0.7
    assert (pose.box.x1, pose.box.y1, pose.box.x2, pose.box.y2) == (
        10.0,
        20.0,
        30.0,
        50.0,
    )
    assert pose.keypoints == [
        (10.0 + i, 20.0 + i, 0.5 + i / 100.0) for i in range(len(KEYPOINT_NAMES))
    ]


def test_keypoint_columns_are_three_per_name_and_prefixed():
    assert POSE_COLUMNS == [
        "pose_score",
        "pose_x1",
        "pose_y1",
        "pose_x2",
        "pose_y2",
        *[
            f"pose_{name}_{field}"
            for name in KEYPOINT_NAMES
            for field in ("x", "y", "score")
        ],
    ]
    assert not np.isnan(_pose(1).score)


def test_default_model_path_uses_app_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.body_pose.user_cache_path",
        lambda appname, appauthor: tmp_path / appauthor / appname,
    )

    assert default_model_path() == (
        tmp_path / "SSC" / "body-eye-sync" / "models" / DEFAULT_MODEL_NAME
    )


def test_resolve_model_caches_only_default_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.body_pose.user_cache_path",
        lambda appname, appauthor: tmp_path / appauthor / appname,
    )

    resolved = _resolve_model(DEFAULT_MODEL_NAME)

    assert resolved == str(
        tmp_path / "SSC" / "body-eye-sync" / "models" / DEFAULT_MODEL_NAME
    )
    assert (tmp_path / "SSC" / "body-eye-sync" / "models").is_dir()
    assert _resolve_model("custom-pose.pt") == "custom-pose.pt"
    assert _resolve_model(tmp_path / "custom-pose.pt") == str(
        tmp_path / "custom-pose.pt"
    )


def test_detect_in_boxes_offsets_best_pose_from_crop():
    keypoints = np.zeros((2, len(KEYPOINT_NAMES), 2), dtype=float)
    scores = np.zeros((2, len(KEYPOINT_NAMES)), dtype=float)
    keypoints[1, :, 0] = np.arange(len(KEYPOINT_NAMES))
    keypoints[1, :, 1] = np.arange(len(KEYPOINT_NAMES)) + 10.0
    scores[1, :] = 0.75

    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array([[0.0, 0.0, 5.0, 5.0], [1.0, 2.0, 11.0, 22.0]]),
            conf=np.array([0.95, 0.9]),
        ),
        keypoints=SimpleNamespace(xy=keypoints, conf=scores),
    )
    model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    poses = _detect_in_boxes(
        model,
        image,
        [BoundingBox(20.0, 30.0, 60.0, 90.0, 7)],
        width=100,
        height=100,
        conf=0.25,
        device="cpu",
    )

    assert len(poses) == 1
    pose = poses[0]
    assert pose.box == BoundingBox(21.0, 32.0, 31.0, 52.0, 7)
    assert pose.score == 0.9
    assert pose.keypoints[0] == (20.0, 40.0, 0.75)
    assert pose.keypoints[-1] == (36.0, 56.0, 0.75)


def test_detect_body_poses_finds_three_people_in_test_video(
    data_dir, tracked_boxes_by_frame
):
    results = list(
        detect_body_poses(data_dir / "three-people.mp4", tracked_boxes_by_frame)
    )

    # example video has five frames, with three tracked people visible throughout
    assert [result.frame_idx for result in results] == [0, 1, 2, 3, 4]
    assert [len(result.poses) for result in results] == [3, 3, 3, 3, 3]
    assert all(
        len(pose.keypoints) == len(KEYPOINT_NAMES)
        for result in results
        for pose in result.poses
    )

    df = poses_to_dataframe(results)
    assert df["track_id"].nunique() == 3

    all_frames = set(df["frame"])
    frames_per_tracklet = df.groupby("track_id")["frame"].agg(set)
    assert all(frames == all_frames for frames in frames_per_tracklet)
