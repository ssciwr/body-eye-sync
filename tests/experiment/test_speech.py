import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.speech import (
    EMBEDDINGS_FILENAME,
    TURNS_FILENAME,
    WORDS_FILENAME,
    Speech,
)
from body_eye_sync.pipeline.diarization import SpeakerEmbedding, SpeakerSegment
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word


def _turns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1],
            "start": [0.0, 2.5],
            "end": [1.75, 4.0],
            "speaker": [0, 1],
        }
    )


def _diarized(speech: Speech, embeddings_per_speaker: int = 0) -> Speech:
    speech.begin_diarization(embeddings_per_speaker)
    speech.add_diarization_segment(SpeakerSegment(0.0, 1.75, 0))
    speech.add_diarization_segment(SpeakerSegment(2.5, 4.0, 1))
    speech.finish_diarization()
    return speech


def _transcribed(speech: Speech) -> Speech:
    speech.begin_transcription()
    speech.add_transcription_segment(
        TranscriptSegment(
            0.0,
            4.0,
            "hallo welt",
            [Word(0.1, 1.0, "hallo", 0.9), Word(2.6, 3.5, "welt", 0.8)],
        )
    )
    speech.finish_transcription()
    return speech


def _embedded(speech: Speech, *, per_speaker: int = 2) -> Speech:
    _diarized(speech, per_speaker)
    for segment_id, (speaker, duration) in enumerate([(0, 1.75), (1, 1.5)]):
        speech.add_speaker_embedding(
            SpeakerEmbedding(segment_id, speaker, duration, np.full(4, float(speaker)))
        )
    speech.finish_speaker_embeddings()
    return speech


def test_new_speech_is_empty():
    speech = Speech()
    assert speech.data is None
    assert speech.words is None
    assert speech.embeddings is None
    assert speech.segments == []
    assert speech.speakers == []


def test_diarization_builds_the_turn_table():
    speech = _diarized(Speech())

    assert speech.data["segment_id"].tolist() == [0, 1]
    assert speech.data["speaker"].tolist() == [0, 1]
    assert speech.speakers == [0, 1]


def test_diarization_drops_previous_results():
    speech = _transcribed(_diarized(Speech()))

    _diarized(speech)

    assert "text" not in speech.data.columns
    assert speech.words is None


def test_clear_drops_partial_diarization():
    speech = Speech()
    speech.begin_diarization()
    speech.add_diarization_segment(SpeakerSegment(0.0, 1.0, 0))
    speech.clear()

    assert speech.data is None


def test_transcription_merges_text_onto_the_turns():
    speech = _transcribed(_diarized(Speech()))

    assert speech.data["text"].tolist() == ["hallo", "welt"]
    assert speech.words["word"].tolist() == ["hallo", "welt"]
    assert speech.words["speaker"].tolist() == [0, 1]


def test_rerunning_transcription_replaces_the_text():
    speech = _transcribed(_diarized(Speech()))

    speech.begin_transcription()
    speech.add_transcription_segment(
        TranscriptSegment(0.0, 1.0, "neu", [Word(0.1, 1.0, "neu", 0.9)])
    )
    speech.finish_transcription()

    # The text column is rebuilt rather than duplicated by the merge.
    assert speech.data["text"].tolist() == ["neu", ""]
    assert list(speech.data.columns).count("text") == 1


def test_transcription_without_diarization_does_nothing():
    speech = Speech()
    speech.begin_transcription()
    speech.add_transcription_segment(
        TranscriptSegment(0.0, 1.0, "neu", [Word(0.1, 1.0, "neu", 0.9)])
    )
    speech.finish_transcription()

    assert speech.data is None


def test_segments_round_trip_through_the_dataframe():
    speech = _diarized(Speech())

    assert [(s.start, s.end, s.speaker) for s in speech.segments] == [
        (0.0, 1.75, 0),
        (2.5, 4.0, 1),
    ]


def test_a_table_that_is_not_speech_results_is_rejected():
    # Every row of a speech table is one turn, keyed by its segment id.
    with pytest.raises(ValueError, match="no 'segment_id' column"):
        Speech().set_data(pd.DataFrame({"nothing": [1]}))


def test_speaker_embeddings_are_kept_per_speaker():
    speech = _embedded(Speech())

    assert speech.embeddings["speaker"].tolist() == [0, 1]
    assert speech.embeddings["segment_id"].tolist() == [0, 1]
    assert speech.embeddings["duration"].tolist() == [1.75, 1.5]


def test_speaker_embeddings_keep_only_the_longest_turns():
    speech = Speech()
    speech.begin_diarization(1)
    speech.finish_diarization()
    # Three turns by one speaker; only the longest survives.
    for segment_id, duration in enumerate([0.5, 3.0, 1.0]):
        speech.add_speaker_embedding(
            SpeakerEmbedding(segment_id, 0, duration, np.full(4, duration))
        )
    speech.finish_speaker_embeddings()

    assert speech.embeddings["duration"].tolist() == [3.0]
    assert speech.embeddings["segment_id"].tolist() == [1]


def test_no_embeddings_collected_when_none_are_asked_for():
    speech = _diarized(Speech(), 0)
    speech.add_speaker_embedding(SpeakerEmbedding(0, 0, 1.75, np.zeros(4)))
    speech.finish_speaker_embeddings()

    assert speech.embeddings is None


def test_round_trip_through_a_directory(tmp_path):
    speech = _transcribed(_embedded(Speech()))

    speech.save(tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {
        TURNS_FILENAME,
        WORDS_FILENAME,
        EMBEDDINGS_FILENAME,
    }
    loaded = Speech()
    loaded.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.data, speech.data)
    pd.testing.assert_frame_equal(loaded.words, speech.words)
    assert loaded.embeddings["speaker"].tolist() == [0, 1]
    # Vectors survive the fixed-size float16 round trip.
    assert np.asarray(loaded.embeddings["embedding"].iloc[1]).tolist() == [1.0] * 4


def test_saving_removes_the_files_it_has_nothing_for(tmp_path):
    # An earlier run had transcription and embeddings on; this one does not.
    for name in (WORDS_FILENAME, EMBEDDINGS_FILENAME):
        (tmp_path / name).write_bytes(b"stale")

    _diarized(Speech()).save(tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {TURNS_FILENAME}


def test_saving_nothing_clears_the_whole_set(tmp_path):
    _transcribed(_embedded(Speech())).save(tmp_path)

    Speech().save(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_loading_a_directory_without_turns_leaves_it_empty(tmp_path):
    speech = _diarized(Speech())

    speech.load(tmp_path)

    assert speech.data is None


def test_loading_replaces_results_that_are_already_there(tmp_path):
    _diarized(Speech()).save(tmp_path)
    speech = _transcribed(_embedded(Speech()))

    speech.load(tmp_path)

    assert "text" not in speech.data.columns
    assert speech.words is None
    assert speech.embeddings is None
