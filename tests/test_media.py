from body_eye_sync.media import has_audio_stream, load_audio


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
    # A camera that records no audio is not an error, just nothing to diarize.
    assert not has_audio_stream(data_dir / "three-people.mp4")


def test_has_audio_stream_is_false_for_an_unreadable_file(tmp_path):
    unreadable = tmp_path / "broken.mp4"
    unreadable.write_bytes(b"not a video")

    assert not has_audio_stream(unreadable)
