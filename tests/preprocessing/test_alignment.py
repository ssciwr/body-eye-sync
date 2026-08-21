import logging

import numpy as np
import pytest

from body_eye_sync.preprocessing.alignment import (
    LANDMARK_HOP,
    LANDMARK_MIN_VOTES,
    PairOffset,
    align,
    align_media,
    landmark_features,
    landmark_offset,
    solve_offsets,
)
from body_eye_sync.preprocessing.audio import SAMPLE_RATE, load_audio

TEST_HOP = 0.02


def test_landmarks_vote_for_the_same_offset_despite_distractors():
    rng = np.random.default_rng(20)
    hashes = rng.integers(0, 1 << 30, size=200)
    a = np.column_stack((np.arange(200) * 10, hashes))
    b = a[20:].copy()
    b[:, 0] -= 200
    distractors = np.column_stack(
        (rng.integers(0, 2000, size=200), rng.integers(0, 1 << 30, size=200))
    )
    b = np.vstack((b, distractors))

    lag, votes = landmark_offset(a, b)

    assert lag == pytest.approx(200 * LANDMARK_HOP)
    assert votes > LANDMARK_MIN_VOTES


def test_unrelated_long_landmark_tracks_do_not_gain_confidence_from_length():
    rng = np.random.default_rng(21)
    count = 135_000
    a = np.column_stack(
        (rng.integers(0, 135_000, size=count), rng.integers(0, 1 << 23, size=count))
    )
    b = np.column_stack(
        (rng.integers(0, 135_000, size=count), rng.integers(0, 1 << 23, size=count))
    )

    _, votes = landmark_offset(a, b)

    assert votes < LANDMARK_MIN_VOTES


def _landmark_recording(hashes, start, end):
    return np.column_stack((np.arange(end - start), hashes[start:end]))


def test_align_solves_every_offset_against_a_reference():
    rng = np.random.default_rng(3)
    hashes = rng.integers(0, 1 << 30, size=2000)
    features = {
        "a": _landmark_recording(hashes, 0, 1500),
        "b": _landmark_recording(hashes, 100, 1700),
        "c": _landmark_recording(hashes, 300, 1900),
    }

    result = align(features)

    assert result.offsets["a"] == pytest.approx(0.0)
    assert result.offsets["b"] == pytest.approx(100 * LANDMARK_HOP)
    assert result.offsets["c"] == pytest.approx(300 * LANDMARK_HOP)
    assert result.ok
    assert result.residual < LANDMARK_HOP


def test_align_reference_only_shifts_the_whole_timeline():
    rng = np.random.default_rng(4)
    hashes = rng.integers(0, 1 << 30, size=2000)
    features = {
        "a": _landmark_recording(hashes, 0, 1500),
        "b": _landmark_recording(hashes, 100, 1700),
    }

    first = align(features, reference="a")
    second = align(features, reference="b")

    assert first.offsets["a"] - first.offsets["b"] == pytest.approx(
        second.offsets["a"] - second.offsets["b"]
    )
    assert second.offsets["b"] == pytest.approx(0.0)


def test_align_reports_pairs_that_do_not_lock(caplog):
    rng = np.random.default_rng(5)
    hashes = rng.integers(0, 1 << 30, size=1200)
    elsewhere = rng.integers(0, 1 << 30, size=900)
    features = {
        "a": _landmark_recording(hashes, 0, 900),
        "b": _landmark_recording(hashes, 80, 1000),
        "elsewhere": _landmark_recording(elsewhere, 0, 900),
    }

    with caplog.at_level(logging.WARNING):
        result = align(features)

    failed = [r.getMessage() for r in caplog.records if "did not lock" in r.message]
    assert any("'a'" in m and "'elsewhere'" in m for m in failed)
    assert any("'b'" in m and "'elsewhere'" in m for m in failed)
    assert not result.ok
    assert result.offsets["b"] == pytest.approx(80 * LANDMARK_HOP)


def test_solve_offsets_residual_is_zero_when_pairs_agree():
    # a->b 1s and b->c 2s imply a->c 3s; all three measurements are consistent.
    pairs = [
        PairOffset("a", "b", 1.0, 20.0),
        PairOffset("b", "c", 2.0, 20.0),
        PairOffset("a", "c", 3.0, 20.0),
    ]

    result = solve_offsets(["a", "b", "c"], pairs)

    assert result.offsets == pytest.approx({"a": 0.0, "b": 1.0, "c": 3.0})
    assert result.residual == pytest.approx(0.0, abs=1e-9)
    assert result.ok


def test_solve_offsets_residual_exposes_contradictory_pairs():
    # Same as above, but a->c disagrees with going via b by a full second.
    pairs = [
        PairOffset("a", "b", 1.0, 20.0),
        PairOffset("b", "c", 2.0, 20.0),
        PairOffset("a", "c", 4.0, 20.0),
    ]

    result = solve_offsets(["a", "b", "c"], pairs)

    assert result.residual > TEST_HOP
    assert not result.ok


def test_solve_offsets_ignores_pairs_that_did_not_lock():
    pairs = [
        PairOffset("a", "b", 1.0, 20.0),
        PairOffset("a", "c", 99.0, 0.5),  # nonsense, and known to be nonsense
        PairOffset("b", "c", 2.0, 20.0),
    ]

    result = solve_offsets(["a", "b", "c"], pairs)

    # The bad measurement is dropped rather than dragging the others off, and
    # the pairs that did lock still connect every input, so this is a good
    # alignment rather than a partial one.
    assert result.offsets == pytest.approx({"a": 0.0, "b": 1.0, "c": 3.0})
    assert result.unaligned == []
    assert result.ok


def test_disconnected_inputs_do_not_receive_arbitrary_zero_offsets():
    result = solve_offsets(["a", "b", "elsewhere"], [PairOffset("a", "b", 1.0, 20.0)])

    assert result.offsets == pytest.approx({"a": 0.0, "b": 1.0})
    assert result.unaligned == ["elsewhere"]
    assert not result.ok


def test_align_of_nothing():
    assert align({}).offsets == {}


def test_landmarks_align_an_overlap_of_only_a_fraction_of_a_second(data_dir):
    """Distinct spectral events align an overlap far too short to correlate.

    The talking video holds the first 0.6 s of the conversation recording, so
    the true offset between them is zero. A smooth speech envelope cannot find
    that over a fraction of a second -- neighbouring lags score as well as the
    right one and the peak lands seconds away -- whereas a handful of agreeing
    landmark hashes place it to within a frame.
    """
    result = align_media(
        {
            "camera": data_dir / "three-people-talking.mp4",
            "recording": data_dir / "three-people-conversation.opus",
        }
    )

    assert result.offsets["recording"] == pytest.approx(0.0, abs=LANDMARK_HOP)
    assert result.ok


def test_align_media_skips_inputs_with_no_audio(data_dir, caplog):
    result = align_media(
        {
            "camera": data_dir / "three-people-talking.mp4",
            "recording": data_dir / "three-people-conversation.opus",
            "silent": data_dir / "three-people.mp4",
        }
    )

    # A camera that recorded no sound cannot be aligned on it, but must not
    # stop the inputs that can.
    assert "silent" not in result.offsets
    assert set(result.offsets) == {"camera", "recording"}
    assert result.unaligned == ["silent"]
    assert not result.ok
    assert "no audio to align on" in caplog.text


# The glasses fixtures are three views of the conversation recording, each with
# one speaker's turn left loud and the other two pushed down, and each cut to a
# different start. The offset below is how much later than the room recording
# each one begins.
GLASSES = {"glasses-1": 1.20, "glasses-2": 0.40, "glasses-3": 2.10}


def _recordings(data_dir):
    paths = {"room": data_dir / "three-people-conversation.opus"}
    paths.update({name: data_dir / f"three-people-{name}.opus" for name in GLASSES})
    return paths


def test_align_media_recovers_the_offsets_between_recordings(data_dir):
    result = align_media(_recordings(data_dir), reference="room")

    assert result.offsets["room"] == pytest.approx(0.0, abs=TEST_HOP)
    for name, started_later in GLASSES.items():
        assert result.offsets[name] == pytest.approx(started_later, abs=2 * TEST_HOP), (
            name
        )


def test_align_media_locks_every_pair_of_the_experiment(data_dir, caplog):
    with caplog.at_level(logging.WARNING):
        result = align_media(_recordings(data_dir), reference="room")

    # Including glasses against glasses, which share only the quieter copy of
    # each other's speech.
    assert [r for r in caplog.records if "did not lock" in r.message] == []
    assert result.residual < TEST_HOP
    assert result.ok


def test_the_reference_only_shifts_the_experiment_timeline(data_dir):
    paths = _recordings(data_dir)

    room = align_media(paths, reference="room")
    other = align_media(paths, reference="glasses-3")

    assert other.offsets["glasses-3"] == pytest.approx(0.0, abs=TEST_HOP)
    for name in paths:
        gap = room.offsets[name] - room.offsets["glasses-3"]
        assert other.offsets[name] == pytest.approx(gap, abs=2 * TEST_HOP), name


def _loudness(media_path):
    """How loud a recording is over time, in dB, one value per envelope frame."""
    samples = load_audio(media_path, SAMPLE_RATE)
    size = int(round(TEST_HOP * SAMPLE_RATE))
    frames = len(samples) // size
    blocks = samples[: frames * size].reshape(frames, size)
    return 20 * np.log10(np.sqrt((blocks**2).mean(axis=1)) + 1e-6)


def test_each_glasses_fixture_is_loudest_during_its_own_speaker(data_dir):
    """The fixtures are what a microphone worn by each speaker would hear.

    Not an alignment property, but the thing that makes them a fair test of it:
    the recordings differ from one another rather than being copies, so the
    correlation has to work on the pattern they share.
    """
    utterances = [(0.03, 3.14), (3.71, 6.68), (7.20, 10.07)]
    for index, (name, started_later) in enumerate(GLASSES.items()):
        envelope = _loudness(data_dir / f"three-people-{name}.opus")
        levels = []
        for start, end in utterances:
            lo = int((start - started_later) / TEST_HOP)
            hi = int((end - started_later) / TEST_HOP)
            lo, hi = max(lo, 0), min(hi, len(envelope))
            levels.append(envelope[lo:hi].mean() if hi > lo + 10 else -np.inf)
        assert int(np.argmax(levels)) == index, f"{name} should be loudest on {index}"


def test_landmark_features_are_sparse_and_silent_media_has_none(data_dir):
    features = landmark_features(data_dir / "three-people-conversation.opus")

    assert features.ndim == 2
    assert features.shape[1] == 2
    assert len(features) > LANDMARK_MIN_VOTES
    assert len(landmark_features(data_dir / "three-people.mp4")) == 0


def test_align_media_reports_progress(data_dir):
    seen = []
    result = align_media(_recordings(data_dir), reference="room", progress=seen.append)

    assert seen == sorted(seen)  # never goes backwards
    assert 0.0 < seen[0] <= 1.0
    assert seen[-1] == pytest.approx(1.0)
    assert result.ok  # and the answer is unaffected


def test_align_media_gives_up_when_progress_says_to(data_dir):
    calls = []

    def stop(fraction):
        calls.append(fraction)
        return False  # give up on the first report

    result = align_media(_recordings(data_dir), reference="room", progress=stop)

    assert len(calls) == 1
    # Nothing partial: offsets from some of the recordings are not the answer.
    assert result.offsets == {}
