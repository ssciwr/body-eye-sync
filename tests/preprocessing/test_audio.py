import numpy as np
import pytest

from body_eye_sync.preprocessing.audio import has_audio_stream, load_audio


def test_load_audio_decodes_opus(data_dir):
    samples = load_audio(data_dir / "three-people-conversation.opus", 16000)

    assert samples.ndim == 1
    # The fixture is 10.4 seconds long.
    assert 10.0 < len(samples) / 16000 < 11.0


def test_load_audio_decodes_a_video_sound_track(data_dir):
    samples = load_audio(data_dir / "three-people-talking.mp4", 16000)

    # As long as the video it came in, 15 frames at 25 fps.
    assert 0.5 < len(samples) / 16000 < 0.7


def test_has_audio_stream_detects_a_recording(data_dir):
    assert has_audio_stream(data_dir / "three-people-conversation.opus")


def test_has_audio_stream_detects_a_video_sound_track(data_dir):
    assert has_audio_stream(data_dir / "three-people-talking.mp4")


def test_has_audio_stream_is_false_for_a_silent_video(data_dir):
    # A camera that records no audio is not an error, just nothing to align on.
    assert not has_audio_stream(data_dir / "three-people.mp4")


def test_has_audio_stream_is_false_for_an_unreadable_file(tmp_path):
    unreadable = tmp_path / "broken.mp4"
    unreadable.write_bytes(b"not a video")

    assert not has_audio_stream(unreadable)


def _rms(samples: np.ndarray, start: float, end: float, rate: int = 16000) -> float:
    window = samples[round(start * rate) : round(end * rate)]
    return float(np.sqrt(np.mean(np.square(window))))


def test_load_audio_keeps_the_timeline_a_recorder_lost_content_in(
    tmp_path, recording_with_a_lost_buffer
):
    path = recording_with_a_lost_buffer(tmp_path / "dropped-buffer.mp4")

    samples = load_audio(path, 16000)

    # Two seconds of timeline, not the 1.75 seconds of content that survived.
    assert len(samples) / 16000 == pytest.approx(2.0, abs=0.05)


def test_load_audio_returns_silence_where_the_content_never_existed(
    tmp_path, recording_with_a_lost_buffer
):
    path = recording_with_a_lost_buffer(
        tmp_path / "dropped-buffer.mp4", lost_at=1.0, lost=0.25
    )

    samples = load_audio(path, 16000)

    assert _rms(samples, 1.05, 1.20) < 1e-3  # the stretch that was lost
    assert _rms(samples, 0.5, 0.9) > 0.1  # before it
    assert _rms(samples, 1.4, 1.9) > 0.1  # and after, still at its own timestamps
