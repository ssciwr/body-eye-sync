import pandas as pd
import pytest

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.pipeline.diarization import SpeakerSegment
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1],
            "start": [0.0, 2.5],
            "end": [1.75, 4.0],
            "speaker": [0, 1],
        }
    )


def _diarized(audio: Audio) -> Audio:
    audio.begin_diarization()
    audio.add_diarization_segment(SpeakerSegment(0.0, 1.75, 0))
    audio.add_diarization_segment(SpeakerSegment(2.5, 4.0, 1))
    audio.finish_diarization()
    return audio


def _transcribed(audio: Audio) -> Audio:
    audio.begin_transcription()
    audio.add_transcription_segment(
        TranscriptSegment(
            0.0,
            4.0,
            "hallo welt",
            [Word(0.1, 1.0, "hallo", 0.9), Word(2.6, 3.5, "welt", 0.8)],
        )
    )
    audio.finish_transcription()
    return audio


def test_new_audio_is_empty():
    audio = Audio()
    assert audio.audio_path is None
    assert audio.data is None
    assert audio.words is None


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


def test_diarization_builds_the_segment_table():
    audio = _diarized(Audio())

    assert audio.data["segment_id"].tolist() == [0, 1]
    assert audio.data["speaker"].tolist() == [0, 1]
    assert audio.speakers == [0, 1]


def test_diarization_drops_previous_results():
    audio = _transcribed(_diarized(Audio()))

    _diarized(audio)

    assert "text" not in audio.data.columns
    assert audio.words is None


def test_discard_diarization_drops_partial_results():
    audio = Audio()
    audio.begin_diarization()
    audio.add_diarization_segment(SpeakerSegment(0.0, 1.0, 0))
    audio.discard_diarization()

    assert audio.data is None


def test_transcription_merges_text_onto_the_turns():
    audio = _transcribed(_diarized(Audio()))

    assert audio.data["text"].tolist() == ["hallo", "welt"]
    assert audio.words["word"].tolist() == ["hallo", "welt"]
    assert audio.words["speaker"].tolist() == [0, 1]


def test_rerunning_transcription_replaces_the_text():
    audio = _transcribed(_diarized(Audio()))

    audio.begin_transcription()
    audio.add_transcription_segment(
        TranscriptSegment(0.0, 1.0, "neu", [Word(0.1, 1.0, "neu", 0.9)])
    )
    audio.finish_transcription()

    # The text column is rebuilt rather than duplicated by the merge.
    assert audio.data["text"].tolist() == ["neu", ""]
    assert list(audio.data.columns).count("text") == 1


def test_discard_transcription_keeps_the_turns():
    audio = _diarized(Audio())
    audio.begin_transcription()
    audio.add_transcription_segment(
        TranscriptSegment(0.0, 1.0, "neu", [Word(0.1, 1.0, "neu", 0.9)])
    )
    audio.discard_transcription()

    assert audio.data["speaker"].tolist() == [0, 1]
    assert audio.words is None


def test_transcription_without_diarization_does_nothing():
    audio = Audio()
    audio.begin_transcription()
    audio.add_transcription_segment(
        TranscriptSegment(0.0, 1.0, "neu", [Word(0.1, 1.0, "neu", 0.9)])
    )
    audio.finish_transcription()

    assert audio.data is None


def test_segments_round_trip_through_the_dataframe():
    audio = _diarized(Audio())

    assert [(s.start, s.end, s.speaker) for s in audio.segments] == [
        (0.0, 1.75, 0),
        (2.5, 4.0, 1),
    ]


def test_parquet_round_trip(tmp_path):
    audio = Audio()
    audio.set_audio("recordings/p1.wav")
    audio.set_data(_segments())
    audio.to_parquet(tmp_path / "mic1.parquet")

    loaded = Audio.from_parquet(tmp_path / "mic1.parquet")
    pd.testing.assert_frame_equal(loaded.data, _segments())


def test_parquet_round_trip_with_words(tmp_path):
    audio = _transcribed(_diarized(Audio()))
    audio.to_parquet(tmp_path / "mic1.parquet")

    assert (tmp_path / "mic1.words.parquet").exists()
    loaded = Audio.from_parquet(tmp_path / "mic1.parquet")
    pd.testing.assert_frame_equal(loaded.data, audio.data)
    pd.testing.assert_frame_equal(loaded.words, audio.words)


def test_loading_without_a_word_file_leaves_words_empty(tmp_path):
    audio = _diarized(Audio())
    audio.to_parquet(tmp_path / "mic1.parquet")

    loaded = Audio.from_parquet(tmp_path / "mic1.parquet")

    assert loaded.words is None


def test_writing_without_data_raises(tmp_path):
    with pytest.raises(ValueError, match="no data to write"):
        Audio().to_parquet(tmp_path / "mic1.parquet")
