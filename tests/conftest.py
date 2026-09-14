import locale
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def c_numeric_locale():
    """Undo the system locale a QApplication applies, as the application does."""
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


@pytest.fixture
def recording_with_a_lost_buffer():
    """Write a file whose recorder dropped a stretch of audio it never wrote.

    Recorders lose capture buffers under load and record the loss by spending
    the timestamps across it anyway, so the file holds less content than its
    timeline covers. Everything that reads the timestamps stays in sync; only a
    decoder that concatenates frames pulls the rest of the recording earlier.
    """
    from fractions import Fraction

    import av
    import numpy as np

    def write(
        path: Path,
        *,
        seconds: float = 2.0,
        lost_at: float = 1.0,
        lost: float = 0.25,
    ) -> Path:
        rate, block = 48_000, 1024
        with av.open(str(path), "w") as container:
            stream = container.add_stream("aac", rate=rate)
            stream.layout = "mono"
            lost_range = range(round(lost_at * rate), round((lost_at + lost) * rate))
            for start in range(0, round(seconds * rate), block):
                if start in lost_range:
                    continue  # never written, but its timestamps are still spent
                times = np.arange(start, start + block, dtype=np.float32) / rate
                samples = (0.25 * np.sin(2 * np.pi * 440 * times))[np.newaxis, :]
                frame = av.AudioFrame.from_ndarray(
                    samples, format="fltp", layout="mono"
                )
                frame.sample_rate = rate
                frame.pts = start
                frame.time_base = Fraction(1, rate)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return path

    return write
