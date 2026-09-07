import numpy as np
import pytest

from body_eye_sync.experiment.timeline import (
    Shift,
    to_experiment_time,
    to_local_time,
    unobserved,
)
from body_eye_sync.preprocessing.timing_correction import (
    SPECTRAL_COEFFICIENTS,
    SPECTRAL_HOP,
    SPECTRAL_MIN_QUALITY,
    OffsetPoint,
    detect_shifts,
    offset_curve,
    fit_timeline,
    pairwise_offset,
    spectral_features,
)

TEST_MIN_QUALITY = 5.0


def test_pairwise_offset_of_a_shifted_copy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 1))
    shift = 40
    b = a[shift:]  # b starts 40 frames later than a

    lag, quality = pairwise_offset(a, b)

    # b's clock is behind a's by 40 frames, so that much is added to reach a.
    assert lag == pytest.approx(shift * SPECTRAL_HOP)
    assert quality > SPECTRAL_MIN_QUALITY


def test_pairwise_offset_is_antisymmetric():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(400, 1))
    b = a[25:]

    forward, _ = pairwise_offset(a, b)
    backward, _ = pairwise_offset(b, a)

    assert forward == pytest.approx(-backward)


def test_unrelated_signals_do_not_lock():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(800, 1)), rng.normal(size=(800, 1))

    _, quality = pairwise_offset(a, b)

    # Noise correlates with noise no better at one lag than any other.
    assert quality < SPECTRAL_MIN_QUALITY


def test_pairwise_offset_of_empty_features():
    lag, quality = pairwise_offset(np.zeros((0, 1)), np.zeros((10, 1)))

    assert (lag, quality) == (0.0, 0.0)


def test_pairwise_offset_requires_feature_columns():
    with pytest.raises(ValueError, match="two-dimensional"):
        pairwise_offset(np.zeros(10), np.zeros(10))


def _room(seconds, seed=7):
    """Synthetic one-column features that never repeat."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(int(seconds / SPECTRAL_HOP), 1))


def _recording(room, started_later, losses=()):
    """A copy of the room that starts later and is missing pieces.

    ``losses`` are ``(experiment_time, seconds)``; the content there is cut out,
    exactly as a device that stalls never writes it.
    """
    start = int(started_later / SPECTRAL_HOP)
    keep = np.ones(len(room), dtype=bool)
    keep[:start] = False
    for at, seconds in losses:
        lo = int(at / SPECTRAL_HOP)
        keep[lo : lo + int(seconds / SPECTRAL_HOP)] = False
    return room[keep]


def _offset_span(points):
    """How far the measured offset moves over the experiment, in seconds."""
    return float(np.ptp([point.offset for point in points]))


def test_offset_curve_is_flat_for_a_recording_that_keeps_time():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    points = offset_curve(room, other, offset=30.0, min_quality=TEST_MIN_QUALITY)

    assert len(points) > 20
    assert all(p.offset == pytest.approx(30.0, abs=SPECTRAL_HOP) for p in points)
    assert _offset_span(points) < 0.05


def test_offset_curve_spaces_windows_at_half_a_window():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    points = offset_curve(
        room,
        other,
        offset=30.0,
        window=20.0,
        min_quality=TEST_MIN_QUALITY,
    )

    spacing = np.diff([point.time for point in points])
    assert all(gap == pytest.approx(10.0) for gap in spacing)


def test_offset_curve_min_quality_overrides_the_default_threshold():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    # Nothing can stand 1000 sigma above its own correlation curve.
    assert offset_curve(room, other, 30.0, min_quality=1000.0) == []
    assert offset_curve(room, other, 30.0, min_quality=0.0)


def test_offset_curve_follows_a_recording_that_loses_content():
    room = _room(900)
    # Two stalls, 0.4 s and 0.9 s, a few minutes apart.
    other = _recording(room, 20.0, losses=[(300.0, 0.4), (600.0, 0.9)])

    points = offset_curve(room, other, offset=20.0, min_quality=TEST_MIN_QUALITY)

    assert _offset_span(points) == pytest.approx(1.3, abs=0.1)
    first = [p.offset for p in points if p.time < 250]
    last = [p.offset for p in points if p.time > 700]
    assert np.median(first) == pytest.approx(20.0, abs=0.05)
    assert np.median(last) == pytest.approx(21.3, abs=0.1)


def test_detect_shifts_finds_each_loss():
    room = _room(900)
    other = _recording(room, 20.0, losses=[(300.0, 0.4), (600.0, 0.9)])

    shifts = detect_shifts(
        offset_curve(room, other, offset=20.0, min_quality=TEST_MIN_QUALITY)
    )

    assert len(shifts) == 2
    assert [round(s.seconds, 1) for s in shifts] == [0.4, 0.9]
    # Located on the recording's own clock, within a window of the truth.
    assert shifts[0].at == pytest.approx(300.0 - 20.0, abs=60.0)
    assert shifts[1].at == pytest.approx(600.0 - 20.0 - 0.4, abs=60.0)


def test_detect_shifts_finds_nothing_in_a_steady_recording():
    room = _room(600)

    assert (
        detect_shifts(
            offset_curve(
                room,
                _recording(room, 30.0),
                offset=30.0,
                min_quality=TEST_MIN_QUALITY,
            )
        )
        == []
    )


def test_detect_shifts_ignores_noise_below_min_shift():
    points = [OffsetPoint(t, 5.0 + 0.01 * (t % 2)) for t in np.arange(0, 600, 15)]

    assert detect_shifts(points) == []


def test_detect_shifts_needs_a_plateau_either_side():
    # One stray window between two levels is not evidence of two losses.
    points = [OffsetPoint(float(t), 1.0) for t in range(0, 100, 15)]
    points += [OffsetPoint(100.0, 1.5)]
    points += [OffsetPoint(float(t), 2.0) for t in range(115, 300, 15)]

    shifts = detect_shifts(points)

    assert len(shifts) == 1
    assert shifts[0].seconds == pytest.approx(1.0)


def test_detect_shifts_reads_a_ramp_as_one_loss():
    """A loss is measured as a ramp, not a step, and is still one loss.

    A window that straddles the missing content matches no single lag and
    reports something in between, so the offset climbs over several windows.
    Read naively that looks like a run of small losses.
    """
    points = [OffsetPoint(float(t), 1.0) for t in range(0, 200, 10)]
    # The ramp is as wide as the window, which is twice the step between
    # windows, so a loss shows up over about two of them.
    points += [OffsetPoint(200.0, 1.4), OffsetPoint(210.0, 1.8)]
    points += [OffsetPoint(float(t), 2.2) for t in range(220, 420, 10)]

    shifts = detect_shifts(points)

    assert len(shifts) == 1
    assert shifts[0].seconds == pytest.approx(1.2, abs=0.05)


def test_detect_shifts_waits_for_the_level_after_a_gradual_ramp():
    points = [OffsetPoint(float(t), 1.0) for t in range(0, 200, 10)]
    points += [
        OffsetPoint(200.0, 1.03),
        OffsetPoint(210.0, 1.06),
        OffsetPoint(220.0, 1.09),
    ]
    points += [OffsetPoint(float(t), 1.12) for t in range(230, 430, 10)]

    shifts = detect_shifts(points)

    assert len(shifts) == 1
    assert shifts[0].seconds == pytest.approx(0.12)


def test_detect_shifts_still_separates_losses_that_are_far_apart():
    points = [OffsetPoint(float(t), 1.0) for t in range(0, 200, 10)]
    points += [OffsetPoint(float(t), 1.4) for t in range(200, 400, 10)]
    points += [OffsetPoint(float(t), 2.1) for t in range(400, 600, 10)]

    shifts = detect_shifts(points)

    assert [round(s.seconds, 1) for s in shifts] == [0.4, 0.7]


def test_clock_fit_does_not_absorb_a_drop_measured_as_a_short_ramp():
    points = []
    for local in np.arange(0.0, 600.0, 10.0):
        missing = 0.0 if local < 300 else 0.05 if local < 310 else 0.1
        experiment = 20.0 + local + missing
        points.append(OffsetPoint(experiment, experiment - local))

    fit = fit_timeline(points)

    assert [shift.seconds for shift in fit.timeline.shifts] == pytest.approx([0.1])


def test_to_experiment_time_maps_a_local_time_onto_the_experiment():
    shifts = [Shift(at=100.0, seconds=0.4), Shift(at=250.0, seconds=0.9)]

    # Before any loss, only the base offset applies.
    assert to_experiment_time(50.0, 20.0, shifts) == pytest.approx(70.0)
    # After the first, the content missing there has to be accounted for.
    assert to_experiment_time(150.0, 20.0, shifts) == pytest.approx(170.4)
    assert to_experiment_time(300.0, 20.0, shifts) == pytest.approx(321.3)


def test_to_experiment_time_with_no_shifts_is_just_the_offset():
    assert to_experiment_time(123.0, 4.5, []) == pytest.approx(127.5)


def test_shifts_recovered_from_a_curve_reconstruct_the_experiment_clock():
    """The whole point: the numbers detected put the recording back on the clock."""
    room = _room(900)
    other = _recording(room, 20.0, losses=[(300.0, 0.4), (600.0, 0.9)])
    shifts = detect_shifts(
        offset_curve(room, other, offset=20.0, min_quality=TEST_MIN_QUALITY)
    )

    # A moment late in the recording, mapped back to when it happened.
    late_local = 800.0
    experiment = to_experiment_time(late_local, 20.0, shifts)

    # A single offset would have been wrong by the whole 1.3 s of losses.
    assert experiment - late_local == pytest.approx(21.3, abs=0.1)


def test_spectral_features_describe_each_frame_with_several_numbers(data_dir):
    values = spectral_features(data_dir / "three-people-conversation.opus")

    assert values.ndim == 2
    assert values.shape[1] == SPECTRAL_COEFFICIENTS
    # Covering the same 10.4 seconds, at the finer spectral hop.
    assert values.shape[0] * SPECTRAL_HOP == pytest.approx(10.4, rel=0.05)


def test_spectral_features_of_a_silent_video(data_dir):
    assert len(spectral_features(data_dir / "three-people.mp4")) == 0


def test_spectral_features_find_losses():
    rng = np.random.default_rng(11)
    room = rng.normal(size=(int(900 / SPECTRAL_HOP), 8))
    keep = np.ones(len(room), dtype=bool)
    for at, seconds in ((300.0, 0.4), (600.0, 0.9)):
        lo = int(at / SPECTRAL_HOP)
        keep[lo : lo + int(seconds / SPECTRAL_HOP)] = False
    keep[: int(20.0 / SPECTRAL_HOP)] = False

    points = offset_curve(
        room,
        room[keep],
        20.0,
        min_quality=TEST_MIN_QUALITY,
    )
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


def test_offset_curve_reports_progress():
    room = _room(600)
    seen = []

    points = offset_curve(
        room,
        _recording(room, 30.0),
        offset=30.0,
        min_quality=TEST_MIN_QUALITY,
        progress=seen.append,
    )

    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert len(points) > 20  # and the measuring still happened


def test_offset_curve_gives_up_when_progress_says_to():
    room = _room(600)
    calls = []

    def stop(fraction):
        calls.append(fraction)
        return fraction < 0.5  # give up halfway

    points = offset_curve(
        room,
        _recording(room, 30.0),
        offset=30.0,
        min_quality=TEST_MIN_QUALITY,
        progress=stop,
    )

    # Unlike an offset, a partial curve is still worth having: it says whether
    # the part it covers held its time.
    assert points
    assert max(p.time for p in points) < 400
    assert calls[-1] >= 0.5
