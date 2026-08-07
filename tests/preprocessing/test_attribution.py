"""Attribution from levels, on synthetic audio whose speaker is known by construction."""

import wave

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.media import SAMPLE_RATE
from body_eye_sync.preprocessing.alignment import Shift, TimelineFit
from body_eye_sync.preprocessing.attribution import (
    BLEED_AGREEMENT,
    HOP_SECONDS,
    TURN_COLUMNS,
    Levels,
    attribute_segments,
    envelope,
    measure_levels,
)

DURATION = 12.0


def _write(path, samples):
    """Write mono float samples to a 16-bit wav file."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def _speech(spans, gain=1.0, noise=0.001, duration=DURATION, seed=0):
    """Quiet noise throughout, with loud tone bursts over each ``(start, end)``."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, noise, int(duration * SAMPLE_RATE))
    times = np.arange(samples.size) / SAMPLE_RATE
    for start, end in spans:
        span = (times >= start) & (times < end)
        samples[span] += gain * 0.3 * np.sin(2 * np.pi * 220 * times[span])
    return samples


@pytest.fixture
def two_speakers(tmp_path):
    """Two recordings: one loud early, the other loud late, in the other's silence."""
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)], seed=1))
    # Half the gain, so the two are only comparable once each is measured
    # against its own baseline.
    _write(tmp_path / "b.wav", _speech([(6.0, 9.0)], gain=0.5, seed=2))
    return {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}


def _fits(**overrides):
    """Timelines that leave every recording's own clock alone."""
    fits = {name: TimelineFit(0.0, 1.0, []) for name in ("a", "b", "c")}
    fits.update({k: v for k, v in overrides.items() if v is not None})
    return fits


def test_envelope_follows_the_loudness_of_the_recording(tmp_path):
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)]))

    levels = envelope(tmp_path / "a.wav")

    assert levels.size == pytest.approx(DURATION / HOP_SECONDS, abs=1)
    loud = levels[int(2.0 / HOP_SECONDS)]
    quiet = levels[int(10.0 / HOP_SECONDS)]
    assert loud > quiet + 20


def test_envelope_of_a_recording_too_short_to_measure_is_empty(tmp_path):
    _write(tmp_path / "blip.wav", np.zeros(10))

    assert envelope(tmp_path / "blip.wav").size == 0


def test_levels_cover_the_time_every_recording_shares(two_speakers):
    levels = measure_levels(two_speakers, _fits())

    assert levels.ids == ["a", "b"]
    assert levels.above_floor.shape == (2, levels.times.size)
    assert levels.times[0] >= 0.0
    assert levels.times[-1] <= DURATION


def test_the_loudest_recording_is_the_one_that_was_speaking(two_speakers):
    levels = measure_levels(two_speakers, _fits())

    assert levels.share("a", 1.0, 4.0) == pytest.approx(1.0)
    assert levels.share("b", 6.0, 9.0) == pytest.approx(1.0)
    # Each is silent while the other talks, so it wins none of that stretch.
    assert levels.share("b", 1.0, 4.0) == 0.0
    assert levels.share("a", 6.0, 9.0) == 0.0


def test_a_quieter_microphone_still_wins_its_own_speech(two_speakers):
    # "b" was recorded at half the gain of "a" and still owns its own turn,
    # because each recording is measured against its own quiet baseline.
    levels = measure_levels(two_speakers, _fits())

    loudest = levels.loudest()
    speaking = levels.times[(levels.times > 6.5) & (levels.times < 8.5)]
    window = np.isin(levels.times, speaking)
    assert (loudest[window] == levels.ids.index("b")).all()


def test_nobody_is_loudest_while_nobody_is_speaking(two_speakers):
    levels = measure_levels(two_speakers, _fits())

    quiet = (levels.times > 4.5) & (levels.times < 5.5)
    assert (levels.loudest()[quiet] == -1).all()
    assert levels.share("a", 4.5, 5.5) == 0.0


def test_offsets_line_the_recordings_up_before_they_are_compared(tmp_path):
    # The same speech, but "b" was started two seconds before "a".
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)], seed=1))
    _write(tmp_path / "b.wav", _speech([(8.0, 11.0)], gain=0.5, seed=2))
    paths = {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}

    levels = measure_levels(paths, _fits(b=TimelineFit(-2.0, 1.0, [])))

    # On the experiment clock "b" speaks from 6s to 9s, not 8s to 11s, so the
    # stretch where its own clock put the end of that turn is silent.
    assert levels.share("b", 6.0, 9.0) == pytest.approx(1.0)
    assert levels.share("b", 9.5, 11.0) == 0.0
    assert (levels.loudest()[(levels.times > 9.5) & (levels.times < 11.0)] == -1).all()


def test_a_recording_that_lost_content_is_still_placed_correctly(tmp_path):
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)], seed=1))
    _write(tmp_path / "b.wav", _speech([(6.0, 9.0)], gain=0.5, seed=2))
    paths = {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}

    # "b" stalled for a second early on, so everything after sits a second
    # earlier on its own clock than it does in the room.
    levels = measure_levels(paths, _fits(b=TimelineFit(0.0, 1.0, [Shift(2.0, 1.0)])))

    assert levels.share("b", 7.0, 10.0) == pytest.approx(1.0)


def test_recordings_that_do_not_overlap_cannot_be_compared(tmp_path):
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)]))
    _write(tmp_path / "b.wav", _speech([(1.0, 4.0)]))
    paths = {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}

    levels = measure_levels(paths, _fits(b=TimelineFit(1000.0, 1.0, [])))

    assert levels.ids == []
    assert levels.loudest().size == 0


def _transcript(*rows):
    return pd.DataFrame(
        [(index, *row) for index, row in enumerate(rows)],
        columns=["segment_id", "start", "end", "text"],
    )


def test_each_segment_goes_to_the_wearer_who_was_loudest(two_speakers):
    levels = measure_levels(two_speakers, _fits())
    transcripts = {
        # Both microphones heard both speakers, so both transcribed both.
        "a": _transcript((1.0, 4.0, "mine"), (6.0, 9.0, "theirs")),
        "b": _transcript((1.0, 4.0, "theirs"), (6.0, 9.0, "mine")),
    }

    turns = attribute_segments(transcripts, levels, _fits())

    assert list(turns.columns) == TURN_COLUMNS
    assert turns["text"].tolist() == ["mine", "mine"]
    assert turns["speaker"].tolist() == ["a", "b"]
    assert turns["source_segment_id"].tolist() == [0, 1]
    # Sorted by start time, and numbered in that order.
    assert turns["start"].tolist() == [1.0, 6.0]
    assert turns["turn_id"].tolist() == [0, 1]


def test_the_text_comes_from_the_recording_that_won_it(two_speakers):
    levels = measure_levels(two_speakers, _fits())
    transcripts = {
        "a": _transcript((1.0, 4.0, "heard clearly")),
        "b": _transcript((1.0, 4.0, "heard faintly")),
    }

    turns = attribute_segments(transcripts, levels, _fits())

    assert turns["text"].tolist() == ["heard clearly"]
    assert turns["source"].tolist() == ["a"]


def test_a_segment_nobody_owns_is_dropped(two_speakers):
    levels = measure_levels(two_speakers, _fits())
    # Whisper text over a stretch where nobody was speaking loudly.
    transcripts = {"a": _transcript((4.5, 5.5, "invented over silence"))}

    turns = attribute_segments(transcripts, levels, _fits())

    assert turns.empty
    assert list(turns.columns) == TURN_COLUMNS


def test_a_segment_split_between_two_wearers_goes_to_the_larger_share(two_speakers):
    levels = measure_levels(two_speakers, _fits())
    # Spans both turns, but sits mostly inside "b"'s.
    transcripts = {
        "a": _transcript((3.5, 9.0, "mostly theirs")),
        "b": _transcript((3.5, 9.0, "mostly mine")),
    }

    turns = attribute_segments(transcripts, levels, _fits())

    assert turns["speaker"].tolist() == ["b"]
    assert turns["text"].tolist() == ["mostly mine"]


def test_transcripts_are_placed_on_the_experiment_clock(tmp_path):
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0)], seed=1))
    _write(tmp_path / "b.wav", _speech([(8.0, 11.0)], gain=0.5, seed=2))
    paths = {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}
    fits = _fits(b=TimelineFit(-2.0, 1.0, []))
    levels = measure_levels(paths, fits)

    # "b" timed its own speech from 8s; the experiment puts it at 6s.
    turns = attribute_segments({"b": _transcript((8.0, 11.0, "mine"))}, levels, fits)

    assert turns["start"].tolist() == [6.0]
    assert turns["end"].tolist() == [9.0]


def test_an_input_with_no_levels_contributes_nothing(two_speakers):
    levels = measure_levels(two_speakers, _fits())

    turns = attribute_segments({"c": _transcript((1.0, 4.0, "unknown"))}, levels, {})

    assert turns.empty


def test_empty_levels_attribute_nothing():
    levels = Levels([], np.empty(0), np.empty((0, 0)))

    turns = attribute_segments({"a": _transcript((1.0, 4.0, "x"))}, levels, {})

    assert turns.empty


# --- two people talking at once, against one voice reaching two microphones ---


@pytest.fixture
def talking_over(tmp_path):
    """Both wearers are live at once, from 6s to 9s."""
    _write(tmp_path / "a.wav", _speech([(1.0, 4.0), (6.0, 9.0)], seed=1))
    _write(tmp_path / "b.wav", _speech([(6.0, 9.0)], gain=0.9, seed=2))
    return {"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}


def _words(*rows):
    """A per-word table on the recording's own clock."""
    return pd.DataFrame(
        [
            (0, index, start, end, word, 0.9)
            for index, (start, end, word) in enumerate(rows)
        ],
        columns=["segment_id", "word_index", "start", "end", "word", "score"],
    )


def test_a_second_speaker_is_kept_when_they_say_something_else(talking_over):
    levels = measure_levels(talking_over, _fits())
    transcripts = {
        "a": _transcript((6.0, 9.0, "the pasta was excellent")),
        "b": _transcript((6.0, 9.0, "no I disagree entirely")),
    }
    words = {
        "a": _words((6.2, 6.8, "the"), (7.0, 7.6, "pasta"), (8.0, 8.8, "excellent")),
        "b": _words((6.3, 6.9, "no"), (7.1, 7.7, "disagree"), (8.1, 8.9, "entirely")),
    }

    turns = attribute_segments(transcripts, levels, _fits(), words=words)

    # Both were live and said different things, so both spoke.
    assert sorted(turns["speaker"]) == ["a", "b"]
    assert sorted(turns["text"]) == [
        "no I disagree entirely",
        "the pasta was excellent",
    ]


def test_the_same_voice_in_two_microphones_is_kept_once(talking_over):
    levels = measure_levels(talking_over, _fits())
    # "b" heard "a" speaking and transcribed almost the same words.
    transcripts = {
        "a": _transcript((6.0, 9.0, "the pasta was excellent")),
        "b": _transcript((6.0, 9.0, "the pasta was excellent")),
    }
    words = {
        "a": _words((6.2, 6.8, "the"), (7.0, 7.6, "pasta"), (8.0, 8.8, "excellent")),
        "b": _words((6.2, 6.8, "the"), (7.0, 7.6, "pasta"), (8.0, 8.8, "excellent")),
    }

    turns = attribute_segments(transcripts, levels, _fits(), words=words)

    assert len(turns) == 1
    assert turns["speaker"].tolist() == ["a"]


def test_bleed_is_recognised_across_differently_cut_segments(talking_over):
    levels = measure_levels(talking_over, _fits())
    # Whisper split the same speech differently on each recording, so the
    # segments do not line up even though the words do.
    transcripts = {
        "a": _transcript((6.0, 9.0, "the pasta was excellent really")),
        "b": _transcript((6.5, 9.0, "pasta was excellent")),
    }
    words = {
        "a": _words(
            (6.1, 6.4, "the"),
            (6.6, 7.0, "pasta"),
            (7.2, 7.6, "was"),
            (8.0, 8.6, "excellent"),
        ),
        "b": _words((6.6, 7.0, "pasta"), (7.2, 7.6, "was"), (8.0, 8.6, "excellent")),
    }

    turns = attribute_segments(transcripts, levels, _fits(), words=words)

    # Comparing whole segments would call these different; comparing the words
    # they place in the same stretch of time does not.
    assert turns["speaker"].tolist() == ["a"]


def test_a_microphone_that_heard_nothing_of_its_own_is_not_a_speaker(two_speakers):
    levels = measure_levels(two_speakers, _fits())
    # "b" is silent while "a" talks, so whatever text it has there is not its
    # wearer speaking, whatever the words happen to say.
    transcripts = {
        "a": _transcript((1.0, 4.0, "mine")),
        "b": _transcript((1.0, 4.0, "completely different words here")),
    }

    turns = attribute_segments(transcripts, levels, _fits())

    assert turns["speaker"].tolist() == ["a"]


def test_without_word_timings_the_segment_text_is_compared(talking_over):
    levels = measure_levels(talking_over, _fits())
    transcripts = {
        "a": _transcript((6.0, 9.0, "the pasta was excellent")),
        "b": _transcript((6.0, 9.0, "the pasta was excellent")),
    }

    # No word tables at all: the whole segments have to stand in for them.
    turns = attribute_segments(transcripts, levels, _fits())

    assert turns["speaker"].tolist() == ["a"]


def test_how_much_agreement_counts_as_bleed_can_be_chosen(talking_over):
    levels = measure_levels(talking_over, _fits())
    # Three words of four in common: the same speech misheard, or two people
    # echoing each other, depending on where the line is drawn.
    transcripts = {
        "a": _transcript((6.0, 9.0, "the pasta was excellent")),
        "b": _transcript((6.0, 9.0, "the pasta was terrible")),
    }

    lenient = attribute_segments(transcripts, levels, _fits(), agreement=0.9)
    assert sorted(lenient["speaker"]) == ["a", "b"]

    strict = attribute_segments(transcripts, levels, _fits(), agreement=0.5)
    assert strict["speaker"].tolist() == ["a"]


def test_a_default_threshold_is_used_when_none_is_given(talking_over):
    levels = measure_levels(talking_over, _fits())
    transcripts = {
        "a": _transcript((6.0, 9.0, "the pasta was excellent")),
        "b": _transcript((6.0, 9.0, "the pasta was terrible")),
    }

    assert attribute_segments(transcripts, levels, _fits())["speaker"].tolist() == (
        attribute_segments(transcripts, levels, _fits(), agreement=BLEED_AGREEMENT)[
            "speaker"
        ].tolist()
    )


# --- one utterance winning twice ---


def _contested_levels():
    """Levels where "a" is loudest early and "b" late, over one 2-second stretch.

    A segment that spans the changeover can be won by either recording,
    depending on which side of it the segment leans.
    """
    times = np.arange(0.0, 2.0, HOP_SECONDS)
    above = np.zeros((2, times.size))
    half = times.size // 2
    above[0, :half] = 30.0  # "a" is the loud one first
    above[1, half:] = 30.0  # then "b" is
    return Levels(["a", "b"], times, above)


def test_one_utterance_won_by_two_recordings_is_kept_once():
    levels = _contested_levels()
    # Whisper cut the same speech differently, so each recording's version
    # leans onto its own side of the changeover and wins most of itself.
    transcripts = {
        "a": _transcript((0.0, 1.2, "dann können wir auch nicht Lasagne nehmen")),
        "b": _transcript((0.8, 2.0, "können wir auch nicht Lasagne nehmen")),
    }
    assert levels.share("a", 0.0, 1.2) > 0.5
    assert levels.share("b", 0.8, 2.0) > 0.5

    turns = attribute_segments(transcripts, levels, _fits())

    assert len(turns) == 1
    # The recording that won the most of it is the one that heard it best.
    assert turns["speaker"].tolist() == ["a"]


def test_the_louder_copy_of_a_repeated_turn_is_the_one_kept():
    levels = _contested_levels()
    # "b" wins more of its version than "a" does of its own.
    transcripts = {
        "a": _transcript((0.6, 1.2, "können wir auch nicht Lasagne nehmen")),
        "b": _transcript((1.0, 2.0, "können wir auch nicht Lasagne nehmen")),
    }
    assert levels.share("b", 1.0, 2.0) > levels.share("a", 0.6, 1.2) > 0.5

    turns = attribute_segments(transcripts, levels, _fits())

    assert turns["speaker"].tolist() == ["b"]


def test_two_winners_saying_different_things_are_both_kept():
    levels = _contested_levels()
    transcripts = {
        "a": _transcript((0.0, 1.2, "ich mag Käse aber nicht überbacken")),
        "b": _transcript((0.8, 2.0, "das ist doch unglaublich oder")),
    }

    turns = attribute_segments(transcripts, levels, _fits())

    # Overlapping in time but not in what was said: two people, not one.
    assert sorted(turns["speaker"]) == ["a", "b"]


def test_turns_that_do_not_overlap_are_never_treated_as_copies():
    levels = _contested_levels()
    # The same words twice, but at times that do not overlap: somebody
    # repeating themselves, or two people saying the same short thing apart.
    transcripts = {
        "a": _transcript((0.0, 0.4, "genau")),
        "b": _transcript((1.4, 2.0, "genau")),
    }

    turns = attribute_segments(transcripts, levels, _fits())

    assert sorted(turns["speaker"]) == ["a", "b"]


def test_two_copies_that_both_lost_are_still_kept_once():
    """The case that bit on real data: nobody won, and both copies got in.

    A long utterance can be won outright by nobody -- every recording's version
    of it spans other people's speech too -- while several microphones were live
    through it and transcribed it. Judging each copy only against the outright
    winners then lets every copy through, because there is no winner to compare
    them with.
    """
    times = np.arange(0.0, 3.0, HOP_SECONDS)
    above = np.zeros((3, times.size))
    half = times.size // 2
    # "a" and "b" are live throughout -- both microphones hear the utterance --
    # but somebody else is louder for the first half of it, so neither of them
    # is the loudest for enough of their own span to win it outright.
    above[0], above[1] = 20.0, 18.0
    above[2, :half] = 30.0
    above[0, half:], above[1, half:] = 25.0, 20.0
    levels = Levels(["a", "b", "c"], times, above)

    transcripts = {
        "a": _transcript((0.0, 3.0, "dann können wir auch nicht Lasagne nehmen")),
        "b": _transcript((0.0, 3.0, "können wir auch nicht Lasagne nehmen")),
        "c": _transcript((0.0, 1.0, "etwas ganz anderes")),
    }
    # Neither copy wins its own span outright, but both microphones are live.
    assert levels.share("a", 0.0, 3.0) <= 0.5
    assert levels.share("b", 0.0, 3.0) <= 0.5
    assert levels.live_share("a", 0.0, 3.0) > 0.5
    assert levels.live_share("b", 0.0, 3.0) > 0.5

    turns = attribute_segments(transcripts, levels, _fits(a=None, b=None, c=None))

    kept = turns[turns["text"].str.contains("Lasagne")]
    assert len(kept) == 1
    # "a" heard it better than "b" did, so "a" keeps it.
    assert kept["speaker"].tolist() == ["a"]
    # The unrelated turn is untouched.
    assert "etwas ganz anderes" in turns["text"].tolist()
