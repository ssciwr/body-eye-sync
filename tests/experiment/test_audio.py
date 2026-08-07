import pandas as pd

import pytest

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.speech import WORDS_FILENAME
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1],
            "start": [0.0, 2.5],
            "end": [1.75, 4.0],
            "text": ["hallo", "welt"],
        }
    )


def _transcribed(audio: Audio) -> Audio:
    audio.speech.begin_transcription()
    audio.speech.add_transcription_segment(
        TranscriptSegment(
            0.0,
            4.0,
            "hallo welt",
            [Word(0.1, 1.0, "hallo", 0.9), Word(2.6, 3.5, "welt", 0.8)],
        )
    )
    audio.speech.finish_transcription()
    return audio


def _stored(audio: Audio, tmp_path):
    """Save ``audio`` into an output directory, returning that directory."""
    directory = tmp_path / "mic1"
    audio.save(directory)
    return directory


def test_new_audio_is_empty():
    audio = Audio()
    assert audio.audio_path is None
    assert audio.speech.data is None
    assert audio.speech.words is None


def test_clear_keeps_the_path():
    audio = Audio(path="recordings/p1.wav")
    audio.speech.set_data(_segments())
    audio.clear()
    assert audio.speech.data is None
    assert audio.audio_path.name == "p1.wav"


def test_results_are_available_through_speech():
    audio = Audio()
    audio.speech.set_data(_segments())

    assert audio.speech.data["text"].tolist() == ["hallo", "welt"]
    assert audio.speech.text == "hallo welt"


def test_results_are_found_in_an_output_directory(tmp_path):
    assert not Audio().has_results(tmp_path)

    audio = Audio()
    audio.speech.set_data(_segments())
    audio.save(tmp_path)

    assert Audio().has_results(tmp_path)


def test_round_trip_through_an_output_directory(tmp_path):
    audio = Audio(path="recordings/p1.wav")
    audio.speech.set_data(_segments())

    loaded = Audio.from_directory(_stored(audio, tmp_path))

    pd.testing.assert_frame_equal(loaded.speech.data, _segments())


def test_round_trip_with_words(tmp_path):
    audio = Audio()
    audio.speech.set_data(_segments())
    _transcribed(audio)

    directory = _stored(audio, tmp_path)

    assert (directory / WORDS_FILENAME).exists()
    loaded = Audio.from_directory(directory)
    pd.testing.assert_frame_equal(loaded.speech.data, audio.speech.data)
    pd.testing.assert_frame_equal(loaded.speech.words, audio.speech.words)


def test_loading_without_a_word_file_leaves_words_empty(tmp_path):
    audio = Audio()
    audio.speech.set_data(_segments())

    directory = _stored(audio, tmp_path)

    assert not (directory / WORDS_FILENAME).exists()
    assert Audio.from_directory(directory).speech.words is None


def test_saving_without_data_raises(tmp_path):
    with pytest.raises(ValueError, match="no data to write"):
        Audio().save(tmp_path)


def test_loading_a_directory_without_results_leaves_it_empty(tmp_path):
    assert Audio.from_directory(tmp_path).speech.data is None
