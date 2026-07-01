import numpy as np

from body_eye_sync.pipeline.object_tracking import (
    BoundingBox,
    boxes_from_tracks,
    cached_model_path,
    default_device,
    detect_tracklets,
    tracks_to_dataframe,
)


def test_cached_model_path_routes_bare_names_into_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.object_tracking.user_cache_path",
        lambda appname, appauthor: tmp_path / appauthor / appname,
    )
    models_dir = tmp_path / "SSC" / "body-eye-sync" / "models"

    # Any bare weights name (not just one special-cased default) goes to the cache.
    assert cached_model_path("yolo26m.pt") == str(models_dir / "yolo26m.pt")
    assert cached_model_path("custom-pose.pt") == str(models_dir / "custom-pose.pt")
    assert models_dir.is_dir()


def test_cached_model_path_leaves_explicit_paths_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.object_tracking.user_cache_path",
        lambda appname, appauthor: tmp_path / appauthor / appname,
    )

    # A path with a directory part is used as given.
    explicit = tmp_path / "weights" / "custom-pose.pt"
    assert cached_model_path(explicit) == str(explicit)

    # An existing file in the cwd is not rewritten.
    here = tmp_path / "local.pt"
    here.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert cached_model_path("local.pt") == "local.pt"


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

    assert boxes[0] == BoundingBox(0.0, 0.0, 5.0, 5.0, track_id=1)
    assert {b.track_id for b in boxes} == {1, 2}


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


def test_default_device_matches_available_hardware():
    import torch

    device = default_device()
    if torch.cuda.is_available():
        assert device == "0"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"


def test_detect_tracklets_yields_one_result_per_frame(data_dir):
    frame_indices = []
    for frame in detect_tracklets(data_dir / "three-people.mp4"):
        # Three people are visible throughout, one track row per detection.
        assert len(frame.tracks) == 3
        frame_indices.append(frame.frame_idx)

    # example video has five frames
    assert frame_indices == [1, 2, 3, 4, 5]


def test_detect_tracklets_finds_three_people(data_dir):
    # BoxMOT numbers frames from 1; store them 0-based as the app does.
    df = tracks_to_dataframe(
        (frame.frame_idx - 1, frame.tracks)
        for frame in detect_tracklets(data_dir / "three-people.mp4")
    )

    # example video shows three people, each visible the whole time
    assert df["track_id"].nunique() == 3

    all_frames = set(df["frame"])
    frames_per_tracklet = df.groupby("track_id")["frame"].agg(set)
    assert all(frames == all_frames for frames in frames_per_tracklet)
