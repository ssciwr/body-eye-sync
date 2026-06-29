from body_eye_sync.pipeline.detection import tracks_to_dataframe
from body_eye_sync.pipeline.tracking import default_device, detect_tracklets


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
