import pandas as pd
import pytest

from body_eye_sync.experiment.speech_turns import TURNS_FILENAME, SpeechTurns


def _turns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn_id": [0, 1, 2],
            "start": [0.0, 2.5, 2.8],
            "end": [1.75, 4.0, 3.2],
            "speaker": ["cam1", "cam2", "cam1"],
            "source": ["cam1", "cam2", "cam1"],
            "source_segment_id": [0, 0, 1],
            "text": ["hallo", "wie geht es", "gut"],
        }
    )


def test_new_turns_are_empty():
    turns = SpeechTurns()

    assert turns.data is None
    assert not turns.has_data()
    assert turns.speakers == []


def test_speakers_names_the_inputs_speech_was_attributed_to():
    turns = SpeechTurns(_turns())

    assert turns.speakers == ["cam1", "cam2"]


def test_one_speakers_turns_can_be_read_on_their_own():
    turns = SpeechTurns(_turns())

    assert turns.for_speaker("cam1")["text"].tolist() == ["hallo", "gut"]
    assert turns.for_speaker("nobody").empty


def test_overlapping_turns_are_kept():
    # Two people talking at once are two turns over the same stretch of time.
    turns = SpeechTurns(_turns())

    assert (
        turns.data.loc[1, "start"]
        < turns.data.loc[2, "start"]
        < turns.data.loc[1, "end"]
    )


def test_a_table_missing_columns_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        SpeechTurns(pd.DataFrame({"start": [0.0], "end": [1.0]}))


def test_save_and_load_round_trip(tmp_path):
    SpeechTurns(_turns()).save(tmp_path)

    loaded = SpeechTurns()
    loaded.load(tmp_path)

    assert (tmp_path / TURNS_FILENAME).exists()
    pd.testing.assert_frame_equal(loaded.data, _turns())


def test_load_from_an_empty_directory_leaves_it_empty(tmp_path):
    turns = SpeechTurns(_turns())

    turns.load(tmp_path)

    assert turns.data is None


def test_saving_nothing_removes_a_stale_file(tmp_path):
    SpeechTurns(_turns()).save(tmp_path)

    SpeechTurns().save(tmp_path)

    assert not (tmp_path / TURNS_FILENAME).exists()
