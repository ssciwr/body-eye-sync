from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    TimelineConfig,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.video_grid import (
    OUTPUT_FPS,
    VideoGridCancelled,
    construct_video_grid,
)


def _encode(container, stream, frame) -> None:
    for packet in stream.encode(frame):
        container.mux(packet)


def _tone(frequency: int, start: int, samples: int) -> av.AudioFrame:
    times = np.arange(start, start + samples, dtype=np.float32) / 48_000
    data = (0.125 * np.sin(2 * np.pi * frequency * times))[np.newaxis, :]
    frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="mono")
    frame.sample_rate = 48_000
    frame.pts = start
    frame.time_base = Fraction(1, 48_000)
    return frame


def _make_video(
    path: Path,
    *,
    color: str,
    fps: int,
    frequency: int,
    audio_start: float = 0.0,
) -> None:
    colors = {"red": (255, 0, 0), "blue": (0, 0, 255)}
    pixels = np.empty((48, 64, 3), dtype=np.uint8)
    pixels[:] = colors[color]
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=48_000)
        audio.layout = "mono"
        video_index = 0
        audio_index = round(audio_start * 48_000)
        while video_index < 2 * fps or audio_index < 2 * 48_000:
            if video_index < 2 * fps and (
                audio_index >= 2 * 48_000 or video_index / fps <= audio_index / 48_000
            ):
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = video_index
                frame.time_base = Fraction(1, fps)
                _encode(container, video, frame)
                video_index += 1
            else:
                count = min(1024, 2 * 48_000 - audio_index)
                _encode(container, audio, _tone(frequency, audio_index, count))
                audio_index += count
        _encode(container, video, None)
        _encode(container, audio, None)


def _make_audio(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=48_000)
        stream.layout = "mono"
        for start in range(0, 2 * 48_000, 1024):
            count = min(1024, 2 * 48_000 - start)
            _encode(container, stream, _tone(660, start, count))
        _encode(container, stream, None)


def _make_nonzero_pts_video(path: Path) -> None:
    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels[:, :, 1] = 128
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=50)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(50):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = 100 + index
            frame.time_base = Fraction(1, 50)
            _encode(container, stream, frame)
        _encode(container, stream, None)


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


def test_construct_video_grid_synchronizes_25_and_50_fps_video_and_audio(tmp_path):
    video_25 = tmp_path / "red-25.mp4"
    video_50 = tmp_path / "blue-50.mp4"
    microphone = tmp_path / "mic.wav"
    _make_video(video_25, color="red", fps=25, frequency=440)
    _make_video(video_50, color="blue", fps=50, frequency=880)
    _make_audio(microphone)

    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="red-25", path=video_25),
                FixedVideoInput(
                    id="blue-50",
                    path=video_50,
                    timeline=TimelineConfig(offset=0.5),
                ),
            ],
            audio=[
                AudioInput(
                    id="microphone",
                    path=microphone,
                    timeline=TimelineConfig(offset=0.25),
                )
            ],
        )
    )
    output = tmp_path / "grid.mp4"
    progress = []

    construct_video_grid(
        experiment,
        output,
        columns=2,
        cell_size=(64, 48),
        progress=lambda fraction: progress.append(fraction) or True,
    )

    assert progress[-1] == 1.0

    with av.open(str(output)) as container:
        video_stream = container.streams.video[0]
        assert float(video_stream.average_rate) == pytest.approx(OUTPUT_FPS)
        # One row of two cells covering the union of all three inputs.
        assert (video_stream.width, video_stream.height) == (128, 48)
        assert float(video_stream.duration * video_stream.time_base) == pytest.approx(
            2.5, abs=0.05
        )
        assert len(container.streams.audio) == 3
        assert [
            stream.metadata.get("handler_name") for stream in container.streams.audio
        ] == [
            "red-25",
            "blue-50",
            "microphone",
        ]
        assert [
            bool(stream.disposition & av.stream.Disposition.default)
            for stream in container.streams.audio
        ] == [True, False, False]

    # The 50 fps source starts half a second late and runs to 2.5 s.
    frame = _rgb_frame_at(output, 0.25)
    assert frame[42, 32, 0] > 180  # red source is present
    assert np.max(frame[42, 96]) < 20  # blue source has not started
    frame = _rgb_frame_at(output, 0.75)
    assert frame[42, 96, 2] > 180  # 50 fps source is present
    frame = _rgb_frame_at(output, 1.75)
    assert frame[42, 96, 2] > 180
    frame = _rgb_frame_at(output, 2.4)
    assert np.max(frame[42, 32]) < 20  # 25 fps source has ended
    assert frame[42, 96, 2] > 180  # the later-starting one has not

    red_audio = _audio_samples(output, 0)
    blue_audio = _audio_samples(output, 1)
    microphone_audio = _audio_samples(output, 2)
    assert _rms_at(red_audio, 0.5) > 0.03
    assert _rms_at(red_audio, 2.2) < 0.005  # ended at its own two seconds
    assert _rms_at(blue_audio, 0.25) < 0.005
    assert _rms_at(blue_audio, 0.75) > 0.03
    assert _rms_at(blue_audio, 1.75) > 0.03
    assert _rms_at(blue_audio, 2.4) > 0.03
    assert _frequency_magnitude(blue_audio, 880) > 100
    assert _rms_at(microphone_audio, 0.1) < 0.005
    assert _rms_at(microphone_audio, 0.5) > 0.03


def test_construct_video_grid_selects_inputs_and_appends_merged_audio(tmp_path):
    red = tmp_path / "red.mp4"
    blue = tmp_path / "blue.mp4"
    microphone = tmp_path / "mic.wav"
    _make_video(red, color="red", fps=25, frequency=440)
    _make_video(blue, color="blue", fps=50, frequency=880)
    _make_audio(microphone)
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

    construct_video_grid(
        experiment,
        output,
        input_ids=["red", "microphone"],
        include_merged_audio=True,
        cell_size=(64, 48),
        show_labels=False,
    )

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
        assert [
            bool(stream.disposition & av.stream.Disposition.default)
            for stream in container.streams.audio
        ] == [False, False, True]
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


def test_construct_video_grid_normalizes_nonzero_source_pts_and_needs_no_audio(
    tmp_path,
):
    source = tmp_path / "nonzero-pts.mp4"
    output = tmp_path / "grid.mp4"
    _make_nonzero_pts_video(source)
    experiment = Experiment(
        ExperimentConfig(fixed_videos=[FixedVideoInput(id="camera", path=source)])
    )

    construct_video_grid(
        experiment,
        output,
        cell_size=(64, 48),
        show_labels=False,
    )

    with av.open(str(output)) as container:
        assert len(container.streams.audio) == 0
        stream = container.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(25.0)
        assert float(stream.duration * stream.time_base) == pytest.approx(1.0, abs=0.05)
    frame = _rgb_frame_at(output, 0.5)
    assert frame[24, 32, 1] > 80


def test_construct_video_grid_cancellation_leaves_no_partial_output(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "grid.mp4"
    _make_nonzero_pts_video(source)
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
        )

    assert not output.exists()
    assert list(tmp_path.glob(".grid.*.mp4")) == []


def test_construct_video_grid_holds_its_place_across_a_lost_buffer(
    tmp_path, recording_with_a_lost_buffer
):
    """Content after a loss belongs where its timestamps put it, not later."""
    red = tmp_path / "red.mp4"
    _make_video(red, color="red", fps=25, frequency=440)
    lossy = recording_with_a_lost_buffer(tmp_path / "lossy.mp4")

    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[FixedVideoInput(id="red", path=red)],
            audio=[
                AudioInput(
                    id="lossy",
                    path=lossy,
                    timeline=TimelineConfig(offset=0.5),
                )
            ],
        )
    )
    output = tmp_path / "grid.mp4"

    construct_video_grid(experiment, output, columns=1, cell_size=(64, 48))

    track = _audio_samples(output, 1)
    assert _rms_at(track, 0.25) < 0.005  # before this input starts
    assert _rms_at(track, 0.75) > 0.03  # its first quarter second
    assert _rms_at(track, 1.6) < 0.005  # the stretch it lost, at 0.5 + 1.0
    # Had the loss been treated as a cut, this would have been pushed to 2.25.
    assert _rms_at(track, 2.0) > 0.03


def test_construct_video_grid_keeps_all_audio_that_starts_after_video(tmp_path):
    source = tmp_path / "delayed-audio.mp4"
    _make_video(
        source,
        color="red",
        fps=25,
        frequency=440,
        audio_start=0.5,
    )
    experiment = Experiment(
        ExperimentConfig(fixed_videos=[FixedVideoInput(id="camera", path=source)])
    )
    output = tmp_path / "grid.mp4"

    construct_video_grid(experiment, output, cell_size=(64, 48), show_labels=False)

    track = _audio_samples(output, 0)
    assert _rms_at(track, 0.25) < 0.005
    assert _rms_at(track, 0.75) > 0.03
    assert _rms_at(track, 1.75) > 0.03


def test_construct_video_grid_stretches_a_recording_whose_clock_ran_slow(tmp_path):
    """An exaggerated rate; real ones are tens of parts per million."""
    red = tmp_path / "red.mp4"
    _make_video(red, color="red", fps=25, frequency=440)

    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="red", path=red, timeline=TimelineConfig(rate=1.5))
            ]
        )
    )
    output = tmp_path / "grid.mp4"

    construct_video_grid(experiment, output, columns=1, cell_size=(64, 48))

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
    # Two seconds on its own clock is three of the experiment's.
    assert duration == pytest.approx(3.0, abs=0.05)
    assert _rgb_frame_at(output, 2.5)[42, 32, 0] > 180
    assert _rms_at(_audio_samples(output, 0), 2.5) > 0.03
