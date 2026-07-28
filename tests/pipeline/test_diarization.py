import pytest

from body_eye_sync.pipeline.diarization import (
    SEGMENT_COLUMNS,
    SpeakerSegment,
    diarize,
    load_audio,
    segments_to_dataframe,
)

#: The cheapest embedding model available, so CI downloads as little as possible.
CI_EMBEDDING_MODEL = "nemo_en_speakerverification_speakernet.onnx"


def test_segments_to_dataframe_columns_and_dtypes():
    df = segments_to_dataframe(
        [SpeakerSegment(0.0, 1.5, 0), SpeakerSegment(2.0, 3.0, 1)]
    )

    assert list(df.columns) == SEGMENT_COLUMNS
    assert df["segment_id"].dtype == int
    assert df["speaker"].dtype == int
    assert df["start"].dtype == float
    assert df["speaker"].tolist() == [0, 1]


def test_segments_to_dataframe_sorts_by_start_time():
    df = segments_to_dataframe(
        [SpeakerSegment(5.0, 6.0, 1), SpeakerSegment(0.0, 1.0, 0)]
    )

    assert df["start"].tolist() == [0.0, 5.0]
    # segment_id numbers the sorted turns, so it always counts up from zero.
    assert df["segment_id"].tolist() == [0, 1]


def test_segments_to_dataframe_keeps_overlapping_turns():
    # Simultaneous speech: two speakers talking over each other stay separate rows.
    df = segments_to_dataframe(
        [SpeakerSegment(0.0, 2.0, 0), SpeakerSegment(1.0, 3.0, 1)]
    )

    assert len(df) == 2
    assert df["speaker"].tolist() == [0, 1]


def test_segments_to_dataframe_empty():
    df = segments_to_dataframe([])

    assert list(df.columns) == SEGMENT_COLUMNS
    assert len(df) == 0


def test_load_audio_decodes_opus(data_dir):
    samples = load_audio(data_dir / "three-people-conversation.opus", 16000)

    assert samples.ndim == 1
    # The fixture is 10.4 seconds long.
    assert 10.0 < len(samples) / 16000 < 11.0


@pytest.mark.parametrize("num_speakers", [-1, 2])
def test_diarize_finds_the_speech_turns(data_dir, num_speakers):
    segments = diarize(
        data_dir / "three-people-conversation.opus",
        embedding_model=CI_EMBEDDING_MODEL,
        num_speakers=num_speakers,
    )

    # Three utterances separated by clear pauses.
    assert len(segments) == 3
    starts = [s.start for s in segments]
    assert starts == sorted(starts)
    assert segments[0].start < 1.0
    assert segments[-1].end > 9.0
    # Two of the fixture's three voices are near-identical to a speaker
    # embedding model (0.88 cosine), so they cluster together: assert only that
    # the distinctly different middle voice is told apart from the other two.
    assert len(set(s.speaker for s in segments)) >= 2
    assert segments[1].speaker != segments[0].speaker


def test_diarize_reports_progress_and_can_be_aborted(data_dir):
    seen = []

    def progress(fraction):
        seen.append(fraction)
        return False  # abort on the first callback

    segments = diarize(
        data_dir / "three-people-conversation.opus",
        embedding_model=CI_EMBEDDING_MODEL,
        progress=progress,
    )

    assert seen
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert segments == []
