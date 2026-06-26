from body_eye_sync.tracking import detect_tracklets


def test_detect_tracklets_finds_three_people(data_dir):
    tracklets = detect_tracklets(data_dir / "three-people.mp4")

    # The example video shows three people, each visible the whole time.
    assert len(tracklets) == 3

    all_frames = {d.frame for dets in tracklets.values() for d in dets}
    for detections in tracklets.values():
        frames = {d.frame for d in detections}
        assert frames == all_frames
