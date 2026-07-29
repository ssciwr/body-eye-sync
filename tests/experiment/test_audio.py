import pandas as pd
import pytest

from body_eye_sync.experiment.audio import Audio


def _segments() -> pd.DataFrame:
    return pd.DataFrame({"start": [0.0, 2.5], "end": [1.75, 4.0]})


def test_new_audio_is_empty():
    audio = Audio()
    assert audio.audio_path is None
    assert audio.data is None


def test_set_audio_records_the_path():
    audio = Audio()
    audio.set_audio("recordings/p1.wav")
    assert audio.audio_path.name == "p1.wav"
    assert audio.data is None


def test_set_audio_drops_previous_results():
    audio = Audio()
    audio.set_data(_segments())
    audio.set_audio("recordings/p2.wav")
    assert audio.data is None


def test_clear_keeps_the_path():
    audio = Audio()
    audio.set_audio("recordings/p1.wav")
    audio.set_data(_segments())
    audio.clear()
    assert audio.data is None
    assert audio.audio_path.name == "p1.wav"


def test_parquet_round_trip(tmp_path):
    audio = Audio()
    audio.set_audio("recordings/p1.wav")
    audio.set_data(_segments())
    audio.to_parquet(tmp_path / "mic1.parquet")

    loaded = Audio.from_parquet(tmp_path / "mic1.parquet")
    pd.testing.assert_frame_equal(loaded.data, _segments())


def test_writing_without_data_raises(tmp_path):
    with pytest.raises(ValueError, match="no data to write"):
        Audio().to_parquet(tmp_path / "mic1.parquet")
