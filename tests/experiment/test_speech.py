import pandas as pd
import pytest

from body_eye_sync.experiment.speech import (
    SEGMENTS_FILENAME,
    WORDS_FILENAME,
    Speech,
)
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1],
            "start": [0.0, 2.5],
            "end": [1.75, 4.0],
            "text": ["hallo welt", "wie geht es"],
        }
    )


def _transcribed(speech: Speech) -> Speech:
    speech.begin_transcription()
    speech.add_transcription_segment(
        TranscriptSegment(
            0.0,
            1.75,
            "hallo welt",
            [Word(0.1, 1.0, "hallo", 0.9), Word(1.1, 1.7, "welt", 0.8)],
        )
    )
    speech.add_transcription_segment(
        TranscriptSegment(2.5, 4.0, "wie geht es", [Word(2.6, 3.5, "wie", 0.7)])
    )
    speech.finish_transcription()
    return speech


def test_a_new_speech_has_nothing():
    speech = Speech()

    assert speech.data is None
    assert speech.words is None
    assert speech.text == ""


def test_transcription_fills_the_segments_and_the_words():
    speech = _transcribed(Speech())

    assert speech.data["text"].tolist() == ["hallo welt", "wie geht es"]
    assert speech.data["segment_id"].tolist() == [0, 1]
    assert speech.words["word"].tolist() == ["hallo", "welt", "wie"]
    # Each word records which segment it came from and where in it.
    assert speech.words["segment_id"].tolist() == [0, 0, 1]
    assert speech.words["word_index"].tolist() == [0, 1, 0]


def test_segments_are_numbered_in_the_order_they_were_spoken():
    speech = Speech()
    speech.begin_transcription()
    speech.add_transcription_segment(TranscriptSegment(5.0, 6.0, "spät", []))
    speech.add_transcription_segment(TranscriptSegment(0.0, 1.0, "früh", []))
    speech.finish_transcription()

    assert speech.data["text"].tolist() == ["früh", "spät"]
    assert speech.data["segment_id"].tolist() == [0, 1]


def test_text_joins_the_whole_transcript():
    speech = _transcribed(Speech())

    assert speech.text == "hallo welt wie geht es"


def test_a_second_pass_replaces_the_first():
    speech = _transcribed(Speech())

    speech.begin_transcription()
    assert speech.data is None
    speech.add_transcription_segment(TranscriptSegment(0.0, 1.0, "neu", []))
    speech.finish_transcription()

    assert speech.data["text"].tolist() == ["neu"]
    assert speech.words.empty


def test_an_empty_pass_gives_empty_tables_not_none():
    speech = Speech()
    speech.begin_transcription()
    speech.finish_transcription()

    assert speech.data is not None
    assert speech.data.empty
    assert speech.words.empty


def test_set_data_needs_a_segment_id():
    with pytest.raises(ValueError, match="segment_id"):
        Speech().set_data(pd.DataFrame({"start": [0.0]}))


def test_save_writes_a_file_per_kind(tmp_path):
    _transcribed(Speech()).save(tmp_path)

    assert (tmp_path / SEGMENTS_FILENAME).exists()
    assert (tmp_path / WORDS_FILENAME).exists()


def test_load_restores_what_save_wrote(tmp_path):
    _transcribed(Speech()).save(tmp_path)

    loaded = Speech()
    loaded.load(tmp_path)

    assert loaded.data["text"].tolist() == ["hallo welt", "wie geht es"]
    assert loaded.words["word"].tolist() == ["hallo", "welt", "wie"]


def test_load_from_an_empty_directory_leaves_it_empty(tmp_path):
    speech = Speech()
    speech.set_data(_segments())

    speech.load(tmp_path)

    assert speech.data is None


def test_save_removes_files_for_results_that_are_gone(tmp_path):
    _transcribed(Speech()).save(tmp_path)

    speech = Speech()
    speech.set_data(_segments())  # no words this time
    speech.save(tmp_path)

    assert (tmp_path / SEGMENTS_FILENAME).exists()
    assert not (tmp_path / WORDS_FILENAME).exists()


def test_clear_drops_everything():
    speech = _transcribed(Speech())

    speech.clear()

    assert speech.data is None
    assert speech.words is None
