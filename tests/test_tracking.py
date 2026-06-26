from body_eye_sync.tracking import detect_tracklets


def test_detect_tracklets_finds_three_people(data_dir):
    df = detect_tracklets(data_dir / "three-people.mp4")

    # The example video shows three people, each visible the whole time.
    assert df["track_id"].nunique() == 3

    all_frames = set(df["frame"])
    frames_per_tracklet = df.groupby("track_id")["frame"].agg(set)
    assert all(frames == all_frames for frames in frames_per_tracklet)
