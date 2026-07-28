from body_eye_sync.pipeline.diarization import SpeakerSegment
from body_eye_sync.pipeline.transcription import (
    UNASSIGNED,
    WORD_COLUMNS,
    TranscriptSegment,
    Word,
    assign_words,
    transcribe,
    transcript_to_dataframe,
)

CI_MODEL = "tiny"


def _segment(*words):
    return TranscriptSegment(
        words[0].start, words[-1].end, " ".join(w.word for w in words), list(words)
    )


def test_words_go_to_the_turn_they_overlap():
    turns = [SpeakerSegment(0.0, 2.0, 0), SpeakerSegment(3.0, 5.0, 1)]
    transcript = [_segment(Word(0.1, 0.9, "hallo", 0.9), Word(3.1, 3.9, "welt", 0.8))]

    texts, words = assign_words(transcript, turns)

    assert texts == ["hallo", "welt"]
    assert words["segment_id"].tolist() == [0, 1]
    assert words["speaker"].tolist() == [0, 1]
    assert words["word_index"].tolist() == [0, 0]


def test_word_goes_to_the_turn_it_overlaps_most():
    turns = [SpeakerSegment(0.0, 1.0, 0), SpeakerSegment(1.0, 3.0, 7)]
    # Spans both turns, but sits mostly inside the second.
    transcript = [_segment(Word(0.8, 2.0, "geteilt", 0.9))]

    texts, words = assign_words(transcript, turns)

    assert texts == ["", "geteilt"]
    assert words["speaker"].tolist() == [7]


def test_words_outside_every_turn_are_kept_but_unassigned():
    turns = [SpeakerSegment(0.0, 1.0, 0)]
    transcript = [
        _segment(Word(0.1, 0.9, "echt", 0.9), Word(8.0, 9.0, "erfunden", 0.2))
    ]

    texts, words = assign_words(transcript, turns)

    # Text Whisper invented over silence must not be attributed to a speaker,
    # but is still kept so the transcript loses nothing silently.
    assert texts == ["echt"]
    assert words["word"].tolist() == ["echt", "erfunden"]
    assert words["segment_id"].tolist() == [0, UNASSIGNED]
    assert words["speaker"].tolist() == [0, UNASSIGNED]


def test_turns_nobody_spoke_in_get_empty_text():
    turns = [SpeakerSegment(0.0, 1.0, 0), SpeakerSegment(5.0, 6.0, 1)]
    transcript = [_segment(Word(0.1, 0.9, "hallo", 0.9))]

    texts, _ = assign_words(transcript, turns)

    assert texts == ["hallo", ""]


def test_words_are_numbered_within_their_turn():
    turns = [SpeakerSegment(0.0, 3.0, 0)]
    transcript = [
        _segment(
            Word(0.1, 0.5, "eins", 0.9),
            Word(1.0, 1.5, "zwei", 0.9),
            Word(2.0, 2.5, "drei", 0.9),
        )
    ]

    texts, words = assign_words(transcript, turns)

    assert texts == ["eins zwei drei"]
    assert words["word_index"].tolist() == [0, 1, 2]


def test_assign_words_empty():
    _, words = assign_words([], [])

    assert list(words.columns) == WORD_COLUMNS
    assert len(words) == 0


def test_transcript_to_dataframe_is_mergeable_on_segment_id():
    turns = [SpeakerSegment(0.0, 1.0, 0), SpeakerSegment(2.0, 3.0, 1)]
    transcript = [_segment(Word(0.1, 0.9, "hallo", 0.9))]

    text, _ = transcript_to_dataframe(transcript, turns)

    assert list(text.columns) == ["segment_id", "text"]
    assert text["segment_id"].tolist() == [0, 1]


def test_transcribe_reads_the_recording(data_dir):
    segments = list(
        transcribe(
            data_dir / "three-people-conversation.opus",
            model_name=CI_MODEL,
            language="de",
            device="cpu",
        )
    )

    assert segments
    assert all(s.end > s.start for s in segments)
    assert all(s.words for s in segments)
    text = " ".join(s.text for s in segments).lower()
    # The phrases even the tiny model gets right on this fixture.
    assert "guten morgen" in text
    assert "kaffee" in text
    assert "minuten" in text
