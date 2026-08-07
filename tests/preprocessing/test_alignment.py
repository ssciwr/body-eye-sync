import logging

import numpy as np
import pytest

from body_eye_sync.preprocessing.alignment import (
    ENVELOPE_HOP,
    FEATURE_HOP,
    LANDMARK,
    LANDMARK_HOP,
    LANDMARK_MIN_VOTES,
    MIN_QUALITY,
    SPECTRAL,
    DriftPoint,
    Method,
    PairOffset,
    Shift,
    align,
    align_media,
    to_local_time,
    to_experiment_time,
    unobserved,
    detect_shifts,
    drift_curve,
    estimate_time_scale,
    fit_timeline,
    landmark_features,
    landmark_offset,
    pairwise_offset,
    solve_offsets,
    spectral_features,
)
from body_eye_sync.media import SAMPLE_RATE, load_audio


def test_pairwise_offset_of_a_shifted_copy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    shift = 40
    b = a[shift:]  # b starts 40 frames later than a

    lag, quality = pairwise_offset(a, b)

    # b's clock is behind a's by 40 frames, so that much is added to reach a.
    assert lag == pytest.approx(shift * ENVELOPE_HOP)
    assert quality > MIN_QUALITY


def test_pairwise_offset_is_antisymmetric():
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    b = a[25:]

    forward, _ = pairwise_offset(a, b)
    backward, _ = pairwise_offset(b, a)

    assert forward == pytest.approx(-backward)


def test_unrelated_signals_do_not_lock():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=800), rng.normal(size=800)

    _, quality = pairwise_offset(a, b)

    # Noise correlates with noise no better at one lag than any other.
    assert quality < MIN_QUALITY


def test_pairwise_offset_of_empty_envelopes():
    lag, quality = pairwise_offset(np.zeros(0), np.zeros(10))

    assert (lag, quality) == (0.0, 0.0)


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


def test_landmark_matches_recover_start_offset_despite_clock_skew():
    rng = np.random.default_rng(22)
    local = np.arange(0, 100_000, 20)
    hashes = rng.integers(0, 1 << 30, size=len(local))
    b = np.column_stack((local, hashes))
    a = np.column_stack((np.rint(1.001 * local + 200).astype(int), hashes))

    lag, votes = landmark_offset(a, b)

    assert lag == pytest.approx(200 * LANDMARK_HOP, abs=2 * LANDMARK_HOP)
    assert votes > 0.9 * len(local)


def test_align_solves_every_offset_against_a_reference():
    rng = np.random.default_rng(3)
    room = rng.normal(size=2000)
    # Three recordings of one room, each started at a different moment.
    envelopes = {"a": room[:1500], "b": room[100:1700], "c": room[300:1900]}

    result = align(envelopes)

    assert result.offsets["a"] == pytest.approx(0.0)
    # b and c started later, so more is added to their clocks to reach a's.
    assert result.offsets["b"] == pytest.approx(100 * ENVELOPE_HOP)
    assert result.offsets["c"] == pytest.approx(300 * ENVELOPE_HOP)
    assert result.ok
    assert result.residual < ENVELOPE_HOP


def test_align_reference_only_shifts_the_whole_timeline():
    rng = np.random.default_rng(4)
    room = rng.normal(size=2000)
    envelopes = {"a": room[:1500], "b": room[100:1700]}

    first = align(envelopes, reference="a")
    second = align(envelopes, reference="b")

    # The inputs keep their positions relative to each other.
    assert first.offsets["a"] - first.offsets["b"] == pytest.approx(
        second.offsets["a"] - second.offsets["b"]
    )
    assert second.offsets["b"] == pytest.approx(0.0)


def test_align_reports_pairs_that_do_not_lock(caplog):
    rng = np.random.default_rng(5)
    room = rng.normal(size=1200)
    envelopes = {"a": room[:900], "b": room[80:1000], "elsewhere": rng.normal(size=900)}

    with caplog.at_level(logging.WARNING):
        result = align(envelopes)

    failed = [r.getMessage() for r in caplog.records if "did not lock" in r.message]
    assert any("'a'" in m and "'elsewhere'" in m for m in failed)
    assert any("'b'" in m and "'elsewhere'" in m for m in failed)
    assert not result.ok
    # The pair that did lock is still solved.
    assert result.offsets["b"] == pytest.approx(80 * ENVELOPE_HOP)


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

    assert result.residual > ENVELOPE_HOP
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
# different start. scripts/make_alignment_fixtures.py builds them; the offset
# below is how much later than the room recording each one begins.
GLASSES = {"glasses-1": 1.20, "glasses-2": 0.40, "glasses-3": 2.10}


def _recordings(data_dir):
    paths = {"room": data_dir / "three-people-conversation.opus"}
    paths.update({name: data_dir / f"three-people-{name}.opus" for name in GLASSES})
    return paths


def test_align_media_recovers_the_offsets_between_recordings(data_dir):
    result = align_media(_recordings(data_dir), reference="room")

    assert result.offsets["room"] == pytest.approx(0.0, abs=ENVELOPE_HOP)
    for name, started_later in GLASSES.items():
        assert result.offsets[name] == pytest.approx(
            started_later, abs=2 * ENVELOPE_HOP
        ), name


def test_align_media_locks_every_pair_of_the_experiment(data_dir, caplog):
    with caplog.at_level(logging.WARNING):
        result = align_media(_recordings(data_dir), reference="room")

    # Including glasses against glasses, which share only the quieter copy of
    # each other's speech.
    assert [r for r in caplog.records if "did not lock" in r.message] == []
    assert result.residual < ENVELOPE_HOP
    assert result.ok


def test_the_reference_only_shifts_the_experiment_timeline(data_dir):
    paths = _recordings(data_dir)

    room = align_media(paths, reference="room")
    other = align_media(paths, reference="glasses-3")

    assert other.offsets["glasses-3"] == pytest.approx(0.0, abs=ENVELOPE_HOP)
    for name in paths:
        gap = room.offsets[name] - room.offsets["glasses-3"]
        assert other.offsets[name] == pytest.approx(gap, abs=2 * ENVELOPE_HOP), name


def _loudness(media_path):
    """How loud a recording is over time, in dB, one value per envelope frame."""
    samples = load_audio(media_path, SAMPLE_RATE)
    size = int(round(ENVELOPE_HOP * SAMPLE_RATE))
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
            lo = int((start - started_later) / ENVELOPE_HOP)
            hi = int((end - started_later) / ENVELOPE_HOP)
            lo, hi = max(lo, 0), min(hi, len(envelope))
            levels.append(envelope[lo:hi].mean() if hi > lo + 10 else -np.inf)
        assert int(np.argmax(levels)) == index, f"{name} should be loudest on {index}"


def test_landmark_features_are_sparse_and_silent_media_has_none(data_dir):
    features = landmark_features(data_dir / "three-people-conversation.opus")

    assert features.ndim == 2
    assert features.shape[1] == 2
    assert len(features) > LANDMARK_MIN_VOTES
    assert len(landmark_features(data_dir / "three-people.mp4")) == 0


# --- drift: recordings that lose content partway through ------------------


#: The loudness-envelope settings the drift tests measure against: 20 ms frames,
#: and the lock threshold that was measured for an envelope.
_ENVELOPE = Method("envelope", lambda path: np.zeros(0), ENVELOPE_HOP, 5.0)


def _room(seconds, seed=7):
    """A synthetic room envelope: long, and never repeating itself."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=int(seconds / ENVELOPE_HOP))


def _recording(room, started_later, losses=()):
    """A copy of the room that starts later and is missing pieces.

    ``losses`` are ``(experiment_time, seconds)``; the content there is cut out,
    exactly as a device that stalls never writes it.
    """
    start = int(started_later / ENVELOPE_HOP)
    keep = np.ones(len(room), dtype=bool)
    keep[:start] = False
    for at, seconds in losses:
        lo = int(at / ENVELOPE_HOP)
        keep[lo : lo + int(seconds / ENVELOPE_HOP)] = False
    return room[keep]


def _offset_span(points):
    """How far the measured offset moves over the experiment, in seconds."""
    return float(np.ptp([point.offset for point in points]))


def test_drift_curve_is_flat_for_a_recording_that_keeps_time():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    points = drift_curve(room, other, offset=30.0, method=_ENVELOPE)

    assert len(points) > 20
    assert all(p.offset == pytest.approx(30.0, abs=ENVELOPE_HOP) for p in points)
    assert _offset_span(points) < 0.05


def test_drift_curve_spaces_windows_at_half_a_window_by_default():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    points = drift_curve(room, other, offset=30.0, window=20.0, method=_ENVELOPE)

    spacing = np.diff([point.time for point in points])
    assert all(gap == pytest.approx(10.0) for gap in spacing)


def test_drift_curve_step_can_be_given_explicitly():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    points = drift_curve(
        room, other, offset=30.0, window=20.0, step=4.0, method=_ENVELOPE
    )

    spacing = np.diff([point.time for point in points])
    assert all(gap == pytest.approx(4.0) for gap in spacing)


def test_drift_curve_min_quality_overrides_the_method_threshold():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    # Nothing can stand 1000 sigma above its own correlation curve.
    assert drift_curve(room, other, 30.0, min_quality=1000.0, method=_ENVELOPE) == []
    assert drift_curve(room, other, 30.0, min_quality=0.0, method=_ENVELOPE)


def test_drift_curve_follows_a_recording_that_loses_content():
    room = _room(900)
    # Two stalls, 0.4 s and 0.9 s, a few minutes apart.
    other = _recording(room, 20.0, losses=[(300.0, 0.4), (600.0, 0.9)])

    points = drift_curve(room, other, offset=20.0, method=_ENVELOPE)

    assert _offset_span(points) == pytest.approx(1.3, abs=0.1)
    first = [p.offset for p in points if p.time < 250]
    last = [p.offset for p in points if p.time > 700]
    assert np.median(first) == pytest.approx(20.0, abs=0.05)
    assert np.median(last) == pytest.approx(21.3, abs=0.1)


def test_detect_shifts_finds_each_loss():
    room = _room(900)
    other = _recording(room, 20.0, losses=[(300.0, 0.4), (600.0, 0.9)])

    shifts = detect_shifts(drift_curve(room, other, offset=20.0, method=_ENVELOPE))

    assert len(shifts) == 2
    assert [round(s.seconds, 1) for s in shifts] == [0.4, 0.9]
    # Located on the recording's own clock, within a window of the truth.
    assert shifts[0].at == pytest.approx(300.0 - 20.0, abs=60.0)
    assert shifts[1].at == pytest.approx(600.0 - 20.0 - 0.4, abs=60.0)


def test_detect_shifts_finds_nothing_in_a_steady_recording():
    room = _room(600)

    assert (
        detect_shifts(
            drift_curve(room, _recording(room, 30.0), offset=30.0, method=_ENVELOPE)
        )
        == []
    )


def test_detect_shifts_ignores_noise_below_min_shift():
    points = [DriftPoint(t, 5.0 + 0.01 * (t % 2), 20.0) for t in np.arange(0, 600, 15)]

    assert detect_shifts(points) == []


def test_detect_shifts_needs_a_plateau_either_side():
    # One stray window between two levels is not evidence of two losses.
    points = [DriftPoint(float(t), 1.0, 20.0) for t in range(0, 100, 15)]
    points += [DriftPoint(100.0, 1.5, 20.0)]
    points += [DriftPoint(float(t), 2.0, 20.0) for t in range(115, 300, 15)]

    shifts = detect_shifts(points, plateau=2)

    assert len(shifts) == 1
    assert shifts[0].seconds == pytest.approx(1.0)


def test_detect_shifts_reads_a_ramp_as_one_loss():
    """A loss is measured as a ramp, not a step, and is still one loss.

    A window that straddles the missing content matches no single lag and
    reports something in between, so the offset climbs over several windows.
    Read naively that looks like a run of small losses.
    """
    points = [DriftPoint(float(t), 1.0, 20.0) for t in range(0, 200, 10)]
    # The ramp is as wide as the window, which is twice the step between
    # windows, so a loss shows up over about two of them.
    points += [DriftPoint(200.0, 1.4, 20.0), DriftPoint(210.0, 1.8, 20.0)]
    points += [DriftPoint(float(t), 2.2, 20.0) for t in range(220, 420, 10)]

    shifts = detect_shifts(points)

    assert len(shifts) == 1
    assert shifts[0].seconds == pytest.approx(1.2, abs=0.05)


def test_detect_shifts_still_separates_losses_that_are_far_apart():
    points = [DriftPoint(float(t), 1.0, 20.0) for t in range(0, 200, 10)]
    points += [DriftPoint(float(t), 1.4, 20.0) for t in range(200, 400, 10)]
    points += [DriftPoint(float(t), 2.1, 20.0) for t in range(400, 600, 10)]

    shifts = detect_shifts(points)

    assert [round(s.seconds, 1) for s in shifts] == [0.4, 0.7]


def _clock_points(offset, scale, losses=()):
    points = []
    for local in np.arange(0.0, 900.0, 10.0):
        missing = sum(seconds for at, seconds in losses if at <= local)
        experiment = offset + scale * local + missing
        points.append(DriftPoint(experiment, experiment - local, 20.0))
    return points


def test_continuous_clock_drift_is_a_rate_not_missing_content():
    points = _clock_points(20.0, 1.0001)

    fit = fit_timeline(points)

    assert estimate_time_scale(points) == pytest.approx(1.0001, abs=1e-9)
    assert fit.offset == pytest.approx(20.0, abs=1e-9)
    assert fit.scale == pytest.approx(1.0001, abs=1e-9)
    assert fit.shifts == []
    assert fit.residual < 1e-9


def test_timeline_fit_can_be_constrained_to_discrete_shifts_only():
    points = _clock_points(20.0, 1.0001)

    fit = fit_timeline(points, time_scale=1.0)

    assert fit.scale == 1.0
    assert fit.shifts == []
    assert fit.offset == pytest.approx(20.0445)
    assert fit.residual > 0.02


def test_a_run_of_small_losses_is_handed_to_the_gap_fit_when_min_shift_drops():
    """The drift estimator and the gap detector share one threshold.

    Ten 30 ms losses over 15 minutes are individually below the default
    ``min_shift``, so nothing calls them gaps and the rate soaks them up as a
    slope instead. Lowering the threshold has to move both halves of the fit
    together, or the losses fall between them and are claimed by neither.
    """
    losses = [(60.0 * n, 0.03) for n in range(1, 11)]
    points = _clock_points(20.0, 1.0, losses)

    absorbed = fit_timeline(points)
    assert absorbed.shifts == []
    assert absorbed.scale > 1.0  # a rate error that is really ten stalls

    found = fit_timeline(points, min_shift=0.02)
    assert [round(shift.seconds, 3) for shift in found.shifts] == [0.03] * 10
    assert found.scale == pytest.approx(1.0, abs=1e-6)


def test_estimate_time_scale_ramp_threshold_follows_min_shift():
    losses = [(60.0 * n, 0.03) for n in range(1, 11)]
    points = _clock_points(20.0, 1.0, losses)

    # At the default the steps look like drift; below them they do not.
    assert estimate_time_scale(points) > 1.0
    assert estimate_time_scale(points, min_shift=0.02) == pytest.approx(1.0, abs=1e-6)


def test_windowed_alignment_recovers_subframe_per_window_clock_drift():
    rng = np.random.default_rng(23)
    reference = rng.normal(size=int(600 / ENVELOPE_HOP))
    expected_scale = 1 / 1.001
    local_frames = int(len(reference) / expected_scale)
    other = np.interp(
        np.arange(local_frames) * expected_scale,
        np.arange(len(reference)),
        reference,
    )

    method = Method(
        "synthetic",
        lambda path: reference if path == "reference" else other,
        ENVELOPE_HOP,
        _ENVELOPE.min_quality,
    )
    points = drift_curve(reference, other, 0.0, method=method)
    fit = fit_timeline(points)

    assert fit.scale == pytest.approx(expected_scale, abs=1e-6)
    assert fit.shifts == []
    assert fit.residual < ENVELOPE_HOP


def test_clock_fit_separates_rate_drift_from_dropped_content():
    points = _clock_points(20.0, 0.9999, losses=[(300.0, 0.4), (600.0, 0.9)])

    fit = fit_timeline(points)

    assert fit.offset == pytest.approx(20.0, abs=0.05)
    assert fit.scale == pytest.approx(0.9999, abs=1e-9)
    assert [shift.seconds for shift in fit.shifts] == pytest.approx([0.4, 0.9])
    assert [shift.at for shift in fit.shifts] == pytest.approx([300.0, 600.0], abs=15)


def test_clock_fit_does_not_absorb_a_drop_measured_as_a_short_ramp():
    points = []
    for local in np.arange(0.0, 600.0, 10.0):
        missing = 0.0 if local < 300 else 0.05 if local < 310 else 0.1
        experiment = 20.0 + 1.0001 * local + missing
        points.append(DriftPoint(experiment, experiment - local, 20.0))

    fit = fit_timeline(points)

    assert fit.scale == pytest.approx(1.0001, abs=1e-9)
    assert [shift.seconds for shift in fit.shifts] == pytest.approx([0.1])


def test_to_experiment_time_maps_a_local_time_onto_the_experiment():
    shifts = [Shift(at=100.0, seconds=0.4), Shift(at=250.0, seconds=0.9)]

    # Before any loss, only the base offset applies.
    assert to_experiment_time(50.0, 20.0, shifts) == pytest.approx(70.0)
    # After the first, the content missing there has to be accounted for.
    assert to_experiment_time(150.0, 20.0, shifts) == pytest.approx(170.4)
    assert to_experiment_time(300.0, 20.0, shifts) == pytest.approx(321.3)


def test_to_experiment_time_with_no_shifts_is_just_the_offset():
    assert to_experiment_time(123.0, 4.5, []) == pytest.approx(127.5)


def test_time_scale_maps_a_free_running_device_clock_in_both_directions():
    experiment = to_experiment_time(3600.0, 20.0, [], scale=1.0001)

    assert experiment == pytest.approx(3620.36)
    assert to_local_time(experiment, 20.0, [], scale=1.0001) == pytest.approx(3600.0)


def test_shifts_recovered_from_a_curve_reconstruct_the_experiment_clock():
    """The whole point: the numbers detected put the recording back on the clock."""
    room = _room(900)
    other = _recording(room, 20.0, losses=[(300.0, 0.4), (600.0, 0.9)])
    shifts = detect_shifts(drift_curve(room, other, offset=20.0, method=_ENVELOPE))

    # A moment late in the recording, mapped back to when it happened.
    late_local = 800.0
    experiment = to_experiment_time(late_local, 20.0, shifts)

    # A single offset would have been wrong by the whole 1.3 s of losses.
    assert experiment - late_local == pytest.approx(21.3, abs=0.1)


# --- the two feature methods ----------------------------------------------


def test_spectral_features_describe_each_frame_with_several_numbers(data_dir):
    values = spectral_features(data_dir / "three-people-conversation.opus")

    assert values.ndim == 2
    assert values.shape[1] == SPECTRAL.extract.__defaults__[0]
    # Covering the same 10.4 seconds, at the finer spectral hop.
    assert values.shape[0] * FEATURE_HOP == pytest.approx(10.4, rel=0.05)


def test_spectral_features_of_a_silent_video(data_dir):
    assert len(spectral_features(data_dir / "three-people.mp4")) == 0


@pytest.mark.parametrize("method", [SPECTRAL, LANDMARK], ids=lambda m: m.name)
def test_both_methods_recover_the_offsets(data_dir, method):
    result = align_media(_recordings(data_dir), reference="room", method=method)

    for name, started_later in GLASSES.items():
        assert result.offsets[name] == pytest.approx(
            started_later, abs=2 * ENVELOPE_HOP
        ), f"{method.name}/{name}"
    assert result.ok


@pytest.mark.parametrize("method", [_ENVELOPE, SPECTRAL], ids=lambda m: m.name)
def test_both_feature_widths_find_the_same_losses(method):
    """Whatever the features, the drift curve and the shifts agree.

    One value per frame and several per frame go down different branches of the
    correlation, so both shapes are measured against the same losses.
    """
    rng = np.random.default_rng(11)
    single = method is _ENVELOPE
    room = rng.normal(size=(int(900 / method.hop), 1 if single else 8))
    if single:
        room = room[:, 0]
    keep = np.ones(len(room), dtype=bool)
    for at, seconds in ((300.0, 0.4), (600.0, 0.9)):
        lo = int(at / method.hop)
        keep[lo : lo + int(seconds / method.hop)] = False
    keep[: int(20.0 / method.hop)] = False

    points = drift_curve(room, room[keep], 20.0, method=method)
    shifts = detect_shifts(points)

    assert _offset_span(points) == pytest.approx(1.3, abs=0.1)
    assert [round(s.seconds, 1) for s in shifts] == [0.4, 0.9]


def test_to_local_time_inverts_to_experiment_time():
    shifts = [Shift(at=100.0, seconds=0.4), Shift(at=250.0, seconds=0.9)]

    for local in (0.0, 50.0, 99.9, 100.0, 150.0, 249.0, 250.0, 400.0):
        experiment = to_experiment_time(local, 20.0, shifts)
        assert to_local_time(experiment, 20.0, shifts) == pytest.approx(local), local


def test_to_local_time_is_none_where_the_recording_lost_content():
    shifts = [Shift(at=100.0, seconds=0.4)]

    # The experiment ran on through the loss; this recording has nothing for it.
    assert to_local_time(120.2, 20.0, shifts) is None
    # Either side of it there is something.
    assert to_local_time(119.9, 20.0, shifts) == pytest.approx(99.9)
    assert to_local_time(120.5, 20.0, shifts) == pytest.approx(100.1)


def test_unobserved_reports_a_span_per_loss():
    shifts = [Shift(at=100.0, seconds=0.4), Shift(at=250.0, seconds=0.9)]

    spans = unobserved(20.0, shifts)

    assert spans == [
        pytest.approx((120.0, 120.4)),
        pytest.approx((270.4, 271.3)),
    ]
    # Each span is exactly as long as the content that went missing.
    assert [round(b - a, 2) for a, b in spans] == [0.4, 0.9]


def test_a_recording_that_kept_time_has_nothing_unobserved():
    assert unobserved(20.0, []) == []


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


def test_drift_curve_reports_progress():
    room = _room(600)
    seen = []

    points = drift_curve(
        room,
        _recording(room, 30.0),
        offset=30.0,
        method=_ENVELOPE,
        progress=seen.append,
    )

    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert len(points) > 20  # and the measuring still happened


def test_drift_curve_gives_up_when_progress_says_to():
    room = _room(600)
    calls = []

    def stop(fraction):
        calls.append(fraction)
        return fraction < 0.5  # give up halfway

    points = drift_curve(
        room, _recording(room, 30.0), offset=30.0, method=_ENVELOPE, progress=stop
    )

    # Unlike an offset, a partial curve is still worth having: it says whether
    # the part it covers held its time.
    assert points
    assert max(p.time for p in points) < 400
    assert calls[-1] >= 0.5
