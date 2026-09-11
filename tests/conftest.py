import gzip
import json
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
    import av

    def write(
        path: Path,
        *,
        seconds: float = 2.0,
        lost_at: float = 1.0,
        lost: float = 0.25,
    ) -> Path:
        with av.open(str(path), "w") as container:
            stream = container.add_stream("aac", rate=_TONE_RATE)
            stream.layout = "mono"
            lost_range = range(
                round(lost_at * _TONE_RATE), round((lost_at + lost) * _TONE_RATE)
            )
            _write_tone(container, stream, seconds, amplitude=0.25, skip=lost_range)
            for packet in stream.encode(None):
                container.mux(packet)
        return path

    return write


#: Sample rate of the test tone written by :func:`_write_tone`.
_TONE_RATE = 48_000


def _write_tone(
    container,
    stream,
    seconds: float,
    *,
    amplitude: float,
    skip: range = range(0),
) -> None:
    """Encode a 440 Hz mono tone in 1024-sample blocks, each stamped at its start.

    Blocks starting in ``skip`` are never written, though their timestamps are
    still spent, as by a recorder that lost a capture buffer.
    """
    from fractions import Fraction

    import av
    import numpy as np

    rate, block = _TONE_RATE, 1024
    for start in range(0, round(seconds * rate), block):
        if start in skip:
            continue
        times = np.arange(start, start + block, dtype=np.float32) / rate
        samples = (amplitude * np.sin(2 * np.pi * 440 * times))[np.newaxis, :]
        frame = av.AudioFrame.from_ndarray(
            samples.astype(np.float32), format="fltp", layout="mono"
        )
        frame.sample_rate = rate
        frame.pts = start
        frame.time_base = Fraction(1, rate)
        for packet in stream.encode(frame):
            container.mux(packet)


def _write_lines(path: Path, records) -> None:
    """Write records as the gzipped JSON lines the devices store them in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def _g3_sample(timestamp, gaze=None, left=True, right=True):
    """One Glasses 3 gaze record; an eye it could not see is an empty object."""
    if gaze is None:
        return {"type": "gaze", "timestamp": timestamp, "data": {}}

    def eye(offset):
        return {
            "gazeorigin": [30.0 + offset, -14.0, -25.0],
            "gazedirection": [0.0, 0.16, 0.98],
            "pupildiameter": 3.0 + offset,
        }

    return {
        "type": "gaze",
        "timestamp": timestamp,
        "data": {
            "gaze2d": list(gaze),
            "gaze3d": [31.0, 202.0, 1306.0],
            "eyeleft": eye(0.1) if left else {},
            "eyeright": eye(0.2) if right else {},
        },
    }


def _g3_imu(timestamp, accelerometer, gyroscope):
    """One Glasses 3 IMU record: both sensors read at the same moment."""
    return {
        "type": "imu",
        "timestamp": timestamp,
        "data": {"accelerometer": accelerometer, "gyroscope": gyroscope},
    }


@pytest.fixture
def glasses3_recording(tmp_path):
    """Four gaze samples (one invalid, one monocular) and configurable firmware."""

    def build(
        folder: Path | None = None,
        *,
        resolution=(1920, 1080),
        firmware: str | None = "1.14.3+nudelsoppa",
    ) -> Path:
        folder = folder or tmp_path / "20220728T161118Z"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "recording.g3").write_text(
            json.dumps(
                {
                    "uuid": "e49a21d8",
                    "name": "Recording 12",
                    "meta-folder": "meta",
                    "duration": 0.08,
                    "created": "2022-07-28T16:11:18.317077Z",
                    "scenecamera": {
                        "file": "scenevideo.mp4",
                        "camera-calibration": {"resolution": list(resolution)},
                    },
                    "gaze": {"file": "gazedata.gz", "samples": 4, "valid-samples": 3},
                    "imu": {"file": "imudata.gz"},
                }
            )
        )
        meta = folder / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "participant").write_text(json.dumps({"name": "1401"}))
        (meta / "HuSerial").write_text("TG03G-010200450191\n")
        if firmware is not None:
            (meta / "RuVersion").write_text(f"{firmware}\n")
        # Two silent frames: the readers need a scene video, not its content.
        _write_video(
            folder / "scenevideo.mp4", seconds=0.08, size=resolution, sound=False
        )
        _write_lines(
            folder / "gazedata.gz",
            [
                _g3_sample(0.02, (0.25, 0.5)),
                _g3_sample(0.04, (0.5, 0.5), right=False),
                _g3_sample(0.06),
                _g3_sample(0.08, (0.75, 0.25)),
            ],
        )
        # Include an interleaved magnetometer record to test filtering.
        _write_lines(
            folder / "imudata.gz",
            [
                _g3_imu(0.01, [0.0, -9.8, 0.0], [1.0, 2.0, 3.0]),
                {
                    "type": "imu",
                    "timestamp": 0.015,
                    "data": {"magnetometer": [-231.0, 133.0, 278.0]},
                },
                _g3_imu(0.02, [0.1, -9.7, 0.2], [1.5, 2.5, 3.5]),
            ],
        )
        return folder

    return build


@pytest.fixture
def glasses2_recording(tmp_path):
    """Three gaze samples (one invalid) with a device clock mapped by ``vts``."""

    def build(
        folder: Path | None = None,
        *,
        segments=1,
        vts=True,
        resolution=(1920, 1080),
    ) -> Path:
        folder = folder or tmp_path / "es6hcql"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "recording.json").write_text(
            json.dumps(
                {
                    "rec_id": folder.name,
                    "rec_participant": "pfl4ojr",
                    "rec_segments": segments,
                    "rec_length": 1,
                    "rec_et_samples": 3,
                    "rec_et_valid_samples": 2,
                }
            )
        )
        (folder / "participant.json").write_text(
            json.dumps({"pa_id": "pfl4ojr", "pa_info": {"Name": "1403"}})
        )
        (folder / "sysinfo.json").write_text(
            json.dumps({"hu_serial": "TG02G-010129906823"})
        )
        start = 337_748_295
        records = [
            # Gaze records share a sample index; s=0 marks valid readings.
            {
                "ts": start,
                "s": 0,
                "gidx": 1401,
                "pc": [25.8, -24.5, -27.9],
                "eye": "left",
            },
            {"ts": start, "s": 0, "gidx": 1401, "pd": 3.96, "eye": "left"},
            {
                "ts": start,
                "s": 0,
                "gidx": 1401,
                "gd": [-0.019, 0.288, 0.957],
                "eye": "left",
            },
            {"ts": start, "s": 0, "gidx": 1401, "pd": 4.12, "eye": "right"},
            {"ts": start, "s": 0, "gidx": 1401, "l": 293849, "gp": [0.5385, 0.2598]},
            {"ts": start, "s": 0, "gidx": 1401, "gp3": [-7.41, 80.96, 348.42]},
            {"ts": start, "s": 0, "ac": [0.392, -9.821, 1.256]},
            {"ts": start + 10_000, "s": 0, "gy": [-1.19, 3.316, -7.518]},
            {"ts": start + 20_000, "s": 0, "ac": [0.4, -9.8, 1.3]},
            {"ts": start + 20_000, "s": 1, "gidx": 1402, "pd": 0.0, "eye": "left"},
            {"ts": start + 20_000, "s": 1, "gidx": 1402, "l": 293850, "gp": [0.0, 0.0]},
            {"ts": start + 20_000, "s": 1, "gidx": 1402, "gp3": [0.0, 0.0, 0.0]},
            {"ts": start + 40_000, "s": 0, "gidx": 1403, "pd": 3.9, "eye": "left"},
            {"ts": start + 40_000, "s": 0, "gidx": 1403, "pd": 4.1, "eye": "right"},
            {"ts": start + 40_000, "s": 0, "gidx": 1403, "l": 293851, "gp": [0.6, 0.3]},
            {"ts": start + 40_000, "s": 0, "dir": "out", "sig": 1},
        ]
        if vts:
            # The video starts 10 ms after the eye tracking does.
            records.insert(0, {"ts": start + 10_000, "s": 0, "vts": 0})
            records.append({"ts": start + 40_000, "s": 0, "vts": 30_000})
        for number in range(1, segments + 1):
            _write_lines(
                folder / "segments" / str(number) / "livedata.json.gz", records
            )
            (folder / "segments" / str(number) / "segment.json").write_text(
                json.dumps({"seg_id": number, "seg_length": 1})
            )
            _write_video(
                folder / "segments" / str(number) / "fullstream.mp4",
                seconds=0.08,
                size=resolution,
                sound=False,
            )
        return folder

    return build


def _write_video(
    path: Path,
    *,
    delay: float = 0.0,
    seconds: float = 1.0,
    size: tuple[int, int] = (64, 48),
    sound: bool = True,
) -> Path:
    """Write a video whose stream starts at ``delay``, with sound from zero."""
    from fractions import Fraction

    import av
    import numpy as np

    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=25)
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p"
        video.time_base = Fraction(1, 1000)
        audio = None
        if sound:
            audio = container.add_stream("aac", rate=_TONE_RATE)
            audio.layout = "mono"
        for index in range(round(seconds * 25)):
            image = np.full((height, width, 3), (index * 8) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = round((delay + index / 25) * 1000)
            frame.time_base = Fraction(1, 1000)
            for packet in video.encode(frame):
                container.mux(packet)
        if audio is not None:
            _write_tone(container, audio, seconds, amplitude=0.2)
        for stream in (video, audio):
            if stream is not None:
                for packet in stream.encode(None):
                    container.mux(packet)
    return path


@pytest.fixture
def video_starting_late():
    """Write a video with a delayed video stream and audio starting at zero."""

    def write(path: Path, *, delay: float = 0.24, seconds: float = 1.0) -> Path:
        return _write_video(path, delay=delay, seconds=seconds)

    return write


@pytest.fixture
def distinct_videos(tmp_path, data_dir):
    """Copy a test video to distinct files for duplicate-input checks."""
    import shutil

    def make(count: int = 2, source: str = "three-people.mp4") -> list[Path]:
        stem = Path(source).stem
        return [
            Path(shutil.copy(data_dir / source, tmp_path / f"{stem}-{number}.mp4"))
            for number in range(1, count + 1)
        ]

    return make
