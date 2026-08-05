import shutil
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    TimeShift,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.export.video_grid import (
    OUTPUT_FPS,
    VideoGridCancelled,
    _observed_spans,
    construct_video_grid,
)
from body_eye_sync.preprocessing.alignment import Shift


def test_observed_spans_apply_scale_offset_and_coincident_losses():
    video = Video(time_offset=10.0)
    video.time_scale = 2.0
    video.time_shifts = [
        # Intentionally unsorted; losses at one instant are combined.
        Shift(at=3.0, seconds=0.5),
        Shift(at=1.0, seconds=0.25),
        Shift(at=1.0, seconds=0.75),
        # A loss at or before local zero moves all observed content.
        Shift(at=-1.0, seconds=0.2),
        # Nothing follows this loss within the supplied stream duration.
        Shift(at=10.0, seconds=5.0),
    ]

    spans = _observed_spans(video, 4.0)

    assert [
        (
            span.local_start,
            span.local_end,
            span.experiment_start,
            span.experiment_end,
        )
        for span in spans
    ] == pytest.approx(
        [
            (0.0, 1.0, 10.2, 12.2),
            (1.0, 3.0, 13.2, 17.2),
            (3.0, 4.0, 17.7, 19.7),
        ]
    )


def test_observed_spans_reject_empty_or_unknown_duration():
    assert _observed_spans(Video(), 0.0) == []
    assert _observed_spans(Video(), float("nan")) == []


def _make_video(
    ffmpeg: str, path: Path, *, color: str, fps: int, frequency: int
) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x48:r={fps}:d=2",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _make_audio(ffmpeg: str, path: Path) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=48000:duration=2",
            str(path),
        ],
        check=True,
    )


def _make_nonzero_pts_video(ffmpeg: str, path: Path) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=64x48:r=50:d=1",
            "-vf",
            "setpts=PTS+2/TB",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _rgb_frame_at(path: Path, timestamp: float) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
    frame = min(
        frames,
        key=lambda value: abs(float(value.pts * value.time_base) - timestamp),
    )
    return frame.to_ndarray(format="rgb24")


def _audio_samples(path: Path, audio_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.audio[audio_index]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=48000)
        frames = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.extend(resampler.resample(frame))
        frames.extend(resampler.resample(None))
    return np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])


def _rms_at(samples: np.ndarray, timestamp: float) -> float:
    start = round((timestamp - 0.05) * 48000)
    end = round((timestamp + 0.05) * 48000)
    window = samples[max(0, start) : min(len(samples), end)]
    return float(np.sqrt(np.mean(np.square(window))))


def _frequency_magnitude(samples: np.ndarray, frequency: int) -> float:
    """Magnitude of one tone in the middle second of a 48 kHz track."""
    window = samples[24_000:72_000] * np.hanning(48_000)
    spectrum = np.abs(np.fft.rfft(window))
    return float(spectrum[frequency])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_construct_video_grid_synchronizes_25_and_50_fps_video_and_audio(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    video_25 = tmp_path / "red-25.mp4"
    video_50 = tmp_path / "blue-50.mp4"
    microphone = tmp_path / "mic.wav"
    _make_video(ffmpeg, video_25, color="red", fps=25, frequency=440)
    _make_video(ffmpeg, video_50, color="blue", fps=50, frequency=880)
    _make_audio(ffmpeg, microphone)

    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="red-25", path=video_25),
                FixedVideoInput(
                    id="blue-50",
                    path=video_50,
                    time_offset=0.5,
                    time_scale=1.25,
                    time_shifts=[TimeShift(at=1.0, seconds=0.5)],
                ),
            ],
            audio=[AudioInput(id="microphone", path=microphone, time_offset=0.25)],
        )
    )
    output = tmp_path / "grid.mp4"
    progress = []

    result = construct_video_grid(
        experiment,
        output,
        columns=2,
        cell_size=(64, 48),
        progress=lambda fraction: progress.append(fraction) or True,
        ffmpeg_path=ffmpeg,
    )

    assert result.path == output
    assert result.experiment_start == pytest.approx(0.0)
    assert result.experiment_end == pytest.approx(3.5, abs=0.03)
    assert result.audio_tracks == ("red-25", "blue-50", "microphone")
    assert (result.columns, result.rows) == (2, 1)
    assert progress[-1] == 1.0

    with av.open(str(output)) as container:
        video_stream = container.streams.video[0]
        assert float(video_stream.average_rate) == pytest.approx(OUTPUT_FPS)
        assert (video_stream.width, video_stream.height) == (128, 48)
        assert len(container.streams.audio) == 3
        assert [
            stream.metadata.get("handler_name") for stream in container.streams.audio
        ] == [
            "red-25",
            "blue-50",
            "microphone",
        ]

    # The 50 fps source starts half a second late and has a half-second loss
    # after its first local second. Its scale stretches that first second to
    # 1.25 experiment seconds, so the loss occupies [1.75, 2.25).
    frame = _rgb_frame_at(output, 0.25)
    assert frame[42, 32, 0] > 180  # red source is present
    assert np.max(frame[42, 96]) < 20  # blue source has not started
    frame = _rgb_frame_at(output, 0.75)
    assert frame[42, 96, 2] > 180  # 50 fps source is present
    frame = _rgb_frame_at(output, 2.0)
    assert np.max(frame[42, 96]) < 20  # its lost stretch is black
    frame = _rgb_frame_at(output, 2.5)
    assert np.max(frame[42, 32]) < 20  # 25 fps source has ended
    assert frame[42, 96, 2] > 180  # 50 fps source resumed after the loss

    red_audio = _audio_samples(output, 0)
    blue_audio = _audio_samples(output, 1)
    microphone_audio = _audio_samples(output, 2)
    assert _rms_at(red_audio, 0.5) > 0.03
    assert _rms_at(red_audio, 2.5) < 0.005
    assert _rms_at(blue_audio, 0.25) < 0.005
    assert _rms_at(blue_audio, 0.75) > 0.03
    assert _rms_at(blue_audio, 2.0) < 0.005
    assert _rms_at(blue_audio, 2.5) > 0.03
    assert _rms_at(microphone_audio, 0.1) < 0.005
    assert _rms_at(microphone_audio, 0.5) > 0.03


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_construct_video_grid_selects_inputs_and_appends_merged_audio(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    red = tmp_path / "red.mp4"
    blue = tmp_path / "blue.mp4"
    microphone = tmp_path / "mic.wav"
    _make_video(ffmpeg, red, color="red", fps=25, frequency=440)
    _make_video(ffmpeg, blue, color="blue", fps=50, frequency=880)
    _make_audio(ffmpeg, microphone)
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="red", path=red),
                FixedVideoInput(id="blue", path=blue),
            ],
            audio=[AudioInput(id="microphone", path=microphone)],
        )
    )
    output = tmp_path / "selected.mp4"

    result = construct_video_grid(
        experiment,
        output,
        input_ids=["red", "microphone"],
        include_merged_audio=True,
        cell_size=(64, 48),
        show_labels=False,
        ffmpeg_path=ffmpeg,
    )

    assert result.audio_tracks == ("red", "microphone", "Merged audio")
    assert (result.columns, result.rows) == (1, 1)
    with av.open(str(output)) as container:
        assert (
            container.streams.video[0].width,
            container.streams.video[0].height,
        ) == (
            64,
            48,
        )
        assert [
            stream.metadata.get("handler_name") for stream in container.streams.audio
        ] == ["red", "microphone", "Merged audio"]
    merged = _audio_samples(output, 2)
    assert _frequency_magnitude(merged, 440) > 100
    assert _frequency_magnitude(merged, 660) > 100
    assert _frequency_magnitude(merged, 880) < 10


def test_construct_video_grid_rejects_unknown_selected_inputs(tmp_path):
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[FixedVideoInput(id="camera", path=tmp_path / "camera.mp4")]
        )
    )

    with pytest.raises(ValueError, match="unknown input ids.*missing"):
        construct_video_grid(
            experiment,
            tmp_path / "grid.mp4",
            input_ids=["camera", "missing"],
        )


def test_construct_video_grid_requires_a_video(tmp_path):
    experiment = Experiment(
        ExperimentConfig(audio=[AudioInput(id="mic", path=tmp_path / "mic.wav")])
    )
    with pytest.raises(ValueError, match="no video inputs"):
        construct_video_grid(experiment, tmp_path / "grid.mp4")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_construct_video_grid_normalizes_nonzero_source_pts_and_needs_no_audio(
    tmp_path,
):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "nonzero-pts.mp4"
    output = tmp_path / "grid.mp4"
    _make_nonzero_pts_video(ffmpeg, source)
    experiment = Experiment(
        ExperimentConfig(fixed_videos=[FixedVideoInput(id="camera", path=source)])
    )

    result = construct_video_grid(
        experiment,
        output,
        cell_size=(64, 48),
        show_labels=False,
        ffmpeg_path=ffmpeg,
    )

    assert result.audio_tracks == ()
    with av.open(str(output)) as container:
        assert len(container.streams.audio) == 0
        stream = container.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(25.0)
        assert float(stream.duration * stream.time_base) == pytest.approx(1.0, abs=0.05)
    frame = _rgb_frame_at(output, 0.5)
    assert frame[24, 32, 1] > 80


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_construct_video_grid_cancellation_leaves_no_partial_output(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "source.mp4"
    output = tmp_path / "grid.mp4"
    _make_nonzero_pts_video(ffmpeg, source)
    experiment = Experiment(
        ExperimentConfig(fixed_videos=[FixedVideoInput(id="camera", path=source)])
    )

    with pytest.raises(VideoGridCancelled):
        construct_video_grid(
            experiment,
            output,
            cell_size=(64, 48),
            show_labels=False,
            progress=lambda _fraction: False,
            ffmpeg_path=ffmpeg,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".grid.*.mp4")) == []
