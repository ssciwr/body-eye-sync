"""Tests for annotated-video output."""

from pathlib import Path

import av
import numpy as np

from postprocessing import visualize_tracklets
from postprocessing.visualize_tracklets import _ChromeCompatibleWriter


def test_chrome_compatible_writer_creates_h264_mp4(tmp_path: Path) -> None:
    output_path = tmp_path / "annotated.mp4"
    writer = _ChromeCompatibleWriter(output_path, fps=29.97, frame_size=(63, 47))
    writer.write(np.zeros((47, 63, 3), dtype=np.uint8))
    writer.release()

    with av.open(str(output_path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))

    assert stream.codec.name == "h264"
    assert stream.pix_fmt == "yuv420p"
    assert (stream.width, stream.height) == (64, 48)
    assert len(frames) == 1


def test_display_video_uses_h264_by_default(tmp_path: Path, monkeypatch) -> None:
    class FakeVideo:
        id = "fake"

        @staticmethod
        def boxes_for_frame(_frame_index: int) -> list:
            return []

    monkeypatch.setattr(visualize_tracklets.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(visualize_tracklets.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(visualize_tracklets.cv2, "waitKey", lambda *args: -1)
    monkeypatch.setattr(
        visualize_tracklets.cv2,
        "getWindowProperty",
        lambda *args: 1,
    )
    monkeypatch.setattr(visualize_tracklets.cv2, "destroyWindow", lambda *args: None)

    output_path = tmp_path / "displayed.mp4"
    source_path = Path(__file__).resolve().parents[2] / "tests/data/three-people.mp4"

    assert visualize_tracklets.display_video(
        source_path,
        FakeVideo(),
        {},
        output_path=output_path,
    )

    with av.open(str(output_path)) as container:
        assert container.streams.video[0].codec.name == "h264"
