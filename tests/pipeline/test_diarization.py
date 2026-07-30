import numpy as np
import pytest

from body_eye_sync.pipeline.diarization import (
    SEGMENT_COLUMNS,
    SpeakerSegment,
    diarize,
    extract_speaker_embeddings,
    segments_to_dataframe,
)

CI_EMBEDDING_MODEL = "nemo_en_speakerverification_speakernet.onnx"

SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0"


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


@pytest.mark.parametrize("num_speakers", [-1, 2])
def test_diarize_finds_the_speech_turns(data_dir, num_speakers):
    segments = diarize(
        data_dir / "three-people-conversation.opus",
        segmentation_model=SEGMENTATION_MODEL,
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
        segmentation_model=SEGMENTATION_MODEL,
        embedding_model=CI_EMBEDDING_MODEL,
        progress=progress,
    )

    assert seen
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert segments == []


def test_extract_speaker_embeddings_gives_one_vector_per_turn(data_dir):
    segments = [SpeakerSegment(0.0, 3.0, 0), SpeakerSegment(4.0, 7.0, 1)]

    embeddings = list(
        extract_speaker_embeddings(
            data_dir / "three-people-conversation.opus",
            segments,
            embedding_model=CI_EMBEDDING_MODEL,
        )
    )

    assert [e.segment_id for e in embeddings] == [0, 1]
    assert [e.speaker for e in embeddings] == [0, 1]
    assert [e.duration for e in embeddings] == [3.0, 3.0]
    # One fixed-size vector each, and the two turns are not the same voice.
    dims = {e.embedding.shape for e in embeddings}
    assert len(dims) == 1
    assert not np.allclose(embeddings[0].embedding, embeddings[1].embedding)


def test_extract_speaker_embeddings_skips_turns_with_too_little_voice(data_dir):
    segments = [SpeakerSegment(0.0, 0.05, 0), SpeakerSegment(1.0, 4.0, 1)]

    embeddings = list(
        extract_speaker_embeddings(
            data_dir / "three-people-conversation.opus",
            segments,
            embedding_model=CI_EMBEDDING_MODEL,
        )
    )

    # The 50 ms turn is dropped; segment_id still refers to the turn it came from.
    assert [e.segment_id for e in embeddings] == [1]


def test_extract_speaker_embeddings_without_turns_does_nothing(data_dir):
    assert (
        list(
            extract_speaker_embeddings(
                data_dir / "three-people-conversation.opus",
                [],
                embedding_model=CI_EMBEDDING_MODEL,
            )
        )
        == []
    )


def test_diarize_reads_a_video_own_audio_track(data_dir):
    segments = diarize(
        data_dir / "three-people-talking.mp4",
        segmentation_model=SEGMENTATION_MODEL,
        embedding_model=CI_EMBEDDING_MODEL,
    )

    assert len(segments) == 1
    # The one turn covers the clip, rather than being an artefact at its edge.
    assert segments[0].start < 0.1
    assert segments[0].end > 0.5
