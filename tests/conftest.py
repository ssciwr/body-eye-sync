from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory containing test data files (videos, etc.)."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tracked_boxes_by_frame(data_dir):
    """Tracked boxes for the three-people fixture video, keyed by 0-based frame."""
    from body_eye_sync.pipeline.object_tracking import (
        boxes_from_tracks,
        detect_tracklets,
    )

    return {
        frame.frame_idx - 1: boxes_from_tracks(frame.tracks)
        for frame in detect_tracklets(data_dir / "three-people.mp4")
    }
