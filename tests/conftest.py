import locale
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def c_numeric_locale():
    """Undo the system locale a QApplication applies, as the application does.

    Creating a QApplication calls ``setlocale(LC_ALL, "")`` once per session and
    leaves it set for every later test, so without this a comma-decimal system
    locale makes FFmpeg reject filter options written with a decimal point.
    """
    locale.setlocale(locale.LC_NUMERIC, "C")


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
