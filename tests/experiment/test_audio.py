import pandas as pd

import pytest

from body_eye_sync.experiment.audio import Audio


def _segments() -> pd.DataFrame:
    return pd.DataFrame({"start": [0.0, 2.5], "end": [1.75, 4.0]})


def _stored(audio: Audio, tmp_path):
    """Save ``audio`` into an output directory, returning that directory."""
    directory = tmp_path / "mic1"
    audio.save(directory)
    return directory


def test_new_audio_is_empty():
    audio = Audio()
    assert audio.audio_path is None
    assert audio.data is None


def test_clear_keeps_the_path():
    audio = Audio(path="recordings/p1.wav")
    audio.set_data(_segments())
    audio.clear()
    assert audio.data is None
    assert audio.audio_path.name == "p1.wav"


def test_results_are_found_in_an_output_directory(tmp_path):
    assert not Audio().has_results(tmp_path)

    audio = Audio()
    audio.set_data(_segments())
    audio.save(tmp_path)

    assert Audio().has_results(tmp_path)


def test_round_trip_through_an_output_directory(tmp_path):
    audio = Audio(path="recordings/p1.wav")
    audio.set_data(_segments())

    loaded = Audio.from_directory(_stored(audio, tmp_path))

    pd.testing.assert_frame_equal(loaded.data, _segments())


def test_saving_without_data_raises(tmp_path):
    with pytest.raises(ValueError, match="no data to write"):
        Audio().save(tmp_path)


def test_loading_a_directory_without_results_leaves_it_empty(tmp_path):
    assert Audio.from_directory(tmp_path).data is None
