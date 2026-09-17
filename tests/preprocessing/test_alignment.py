import logging

import numpy as np
import pytest

from body_eye_sync.preprocessing.alignment import (
    LANDMARK_HOP,
    PairOffset,
    align,
    align_media,
    landmark_features,
    landmark_offset,
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
    assert votes > 0


def _landmark_recording(hashes, start, end):
    return np.column_stack((np.arange(end - start), hashes[start:end]))


def _align_pairs(ids, pairs):
    """Run align() with predefined pair measurements."""
    index = {name: i for i, name in enumerate(ids)}
    measurements = {
        (index[pair.a], index[pair.b]): (pair.lag, pair.quality) for pair in pairs
    }
    envelopes = {name: np.array([i]) for name, i in index.items()}

    def pairwise(a, b, _hop):
        return measurements.get((int(a[0]), int(b[0])), (0.0, 0.0))

    return align(envelopes, pairwise=pairwise)


def test_align_solves_every_relative_offset():
    rng = np.random.default_rng(3)
    hashes = rng.integers(0, 1 << 30, size=2000)
    features = {
        "a": _landmark_recording(hashes, 0, 1500),
        "b": _landmark_recording(hashes, 100, 1700),
        "c": _landmark_recording(hashes, 300, 1900),
    }

    result = align(features)

    assert result.offsets["b"] - result.offsets["a"] == pytest.approx(
        100 * LANDMARK_HOP
    )
    assert result.offsets["c"] - result.offsets["a"] == pytest.approx(
        300 * LANDMARK_HOP
    )
    assert result.unaligned == []


def test_align_reports_pairs_that_do_not_match(caplog):
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

    failed = [r.getMessage() for r in caplog.records if "did not match" in r.message]
    assert any("'a'" in m and "'elsewhere'" in m for m in failed)
    assert any("'b'" in m and "'elsewhere'" in m for m in failed)
    assert result.unaligned == ["elsewhere"]
    assert result.offsets["b"] == pytest.approx(80 * LANDMARK_HOP)


def test_align_uses_consistent_pairs():
    # a->b 1s and b->c 2s imply a->c 3s; all three measurements are consistent.
    pairs = [
        PairOffset("a", "b", 1.0, 20.0),
        PairOffset("b", "c", 2.0, 20.0),
        PairOffset("a", "c", 3.0, 20.0),
    ]

    result = _align_pairs(["a", "b", "c"], pairs)

    assert result.offsets == pytest.approx({"a": 0.0, "b": 1.0, "c": 3.0})
    assert result.unaligned == []


def test_maximum_spanning_tree_ignores_a_weaker_spurious_candidate():
    # Measurements from the Tobii experiment that exposed the failure. The
    # final pair is a false lock between two non-overlapping parts of 1404.
    pairs = [
        PairOffset("1401", "1402", -74.288, 7914.0),
        PairOffset("1401", "1403", 71.520, 3311.0),
        PairOffset("1401", "1404", 144.704, 503.0),
        PairOffset("1401", "1404_2", 1361.088, 3310.0),
        PairOffset("1402", "1403", 145.808, 3789.0),
        PairOffset("1402", "1404", 218.992, 477.0),
        PairOffset("1402", "1404_2", 1435.376, 3273.0),
        PairOffset("1403", "1404", 73.200, 1229.0),
        PairOffset("1403", "1404_2", 1289.552, 3878.0),
        # A positive-vote candidate, but about 1290 seconds wrong.
        PairOffset("1404", "1404_2", -73.600, 15.0),
    ]

    result = _align_pairs(["1401", "1402", "1403", "1404", "1404_2"], pairs)

    assert result.offsets == pytest.approx(
        {
            "1402": 0.0,
            "1401": 74.288,
            "1403": 145.808,
            "1404": 219.008,
            "1404_2": 1435.360,
        }
    )
    assert result.offsets["1404"] - result.offsets["1401"] == pytest.approx(144.720)
    assert result.unaligned == []


def test_align_ignores_a_weaker_pair_outside_the_tree():
    pairs = [
        PairOffset("a", "b", 1.0, 20.0),
        PairOffset("a", "c", 99.0, 0.5),  # weaker contradictory candidate
        PairOffset("b", "c", 2.0, 20.0),
    ]

    result = _align_pairs(["a", "b", "c"], pairs)

    # The stronger path wins, so the unused pair cannot move the offsets.
    assert result.offsets["b"] - result.offsets["a"] == pytest.approx(1.0)
    assert result.offsets["c"] - result.offsets["a"] == pytest.approx(3.0)
    assert result.unaligned == []


def test_disconnected_inputs_do_not_receive_arbitrary_zero_offsets():
    result = _align_pairs(
        ["a", "b", "elsewhere"],
        [PairOffset("a", "b", 1.0, 20.0), PairOffset("b", "elsewhere", 0.0, 0.0)],
    )

    assert result.offsets == pytest.approx({"a": 0.0, "b": 1.0})
    assert result.unaligned == ["elsewhere"]


def test_align_of_nothing():
    assert align({}).offsets == {}


def test_alignment_can_chain_recordings_when_the_reference_does_not_span_them():
    """A short reference reaches the final recording through the middle one."""
    rng = np.random.default_rng(22)
    hashes = rng.integers(0, 1 << 30, size=3000)
    features = {
        "reference": _landmark_recording(hashes, 0, 1000),
        "middle": _landmark_recording(hashes, 800, 2000),
        "late": _landmark_recording(hashes, 1800, 3000),
    }

    result = align(features)

    assert result.offsets["middle"] - result.offsets["reference"] == pytest.approx(
        800 * LANDMARK_HOP
    )
    assert result.offsets["late"] - result.offsets["reference"] == pytest.approx(
        1800 * LANDMARK_HOP
    )
    assert result.unaligned == []


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
    assert result.unaligned == []


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
    result = align_media(_recordings(data_dir))

    for name, started_later in GLASSES.items():
        relative_offset = result.offsets[name] - result.offsets["room"]
        assert relative_offset == pytest.approx(started_later, abs=2 * TEST_HOP), name


def test_align_media_locks_every_pair_of_the_experiment(data_dir, caplog):
    with caplog.at_level(logging.WARNING):
        result = align_media(_recordings(data_dir))

    # Including glasses against glasses, which share only the quieter copy of
    # each other's speech.
    assert [r for r in caplog.records if "did not match" in r.message] == []
    assert result.unaligned == []


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
    assert len(features) > 0
    assert len(landmark_features(data_dir / "three-people.mp4")) == 0


def test_align_media_reports_progress(data_dir):
    seen = []
    result = align_media(_recordings(data_dir), progress=seen.append)

    assert seen == sorted(seen)  # never goes backwards
    assert 0.0 < seen[0] <= 1.0
    assert seen[-1] == pytest.approx(1.0)
    assert result.unaligned == []  # and the answer is unaffected


def test_align_media_gives_up_when_progress_says_to(data_dir):
    calls = []

    def stop(fraction):
        calls.append(fraction)
        return False  # give up on the first report

    result = align_media(_recordings(data_dir), progress=stop)

    assert len(calls) == 1
    # Nothing partial: offsets from some of the recordings are not the answer.
    assert result.offsets == {}
