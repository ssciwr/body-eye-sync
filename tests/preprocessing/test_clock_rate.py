from dataclasses import replace

import numpy as np
import pytest

from body_eye_sync.experiment.timeline import to_experiment_time, to_local_time
from body_eye_sync.preprocessing.clock_rate import (
    SPECTRAL_COEFFICIENTS,
    SPECTRAL_HOP,
    SPECTRAL_MIN_QUALITY,
    OffsetPoint,
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


def _recording(room, started_later):
    """A copy of the room that starts later but keeps the same time."""
    return room[int(started_later / SPECTRAL_HOP) :]


def _drifting_recording(room, started_later, rate):
    """A copy of the room made on a clock running at ``rate``.

    ``rate`` is experiment seconds per second of this recording's own clock, so
    above one is a device whose clock runs slow and stretches the room out.
    """
    start = int(started_later / SPECTRAL_HOP)
    count = int((len(room) - start) / rate)
    index = start + np.round(np.arange(count) * rate).astype(int)
    return room[index[index < len(room)]]


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


def test_offset_curve_windows_do_not_overlap():
    """Independent windows, so their spread says how well the lag is known."""
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
    assert all(gap == pytest.approx(20.0) for gap in spacing)


def test_offset_curve_min_quality_overrides_the_default_threshold():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    # Nothing can stand 1000 sigma above its own correlation curve.
    assert offset_curve(room, other, 30.0, min_quality=1000.0) == []
    assert offset_curve(room, other, 30.0, min_quality=0.0)


def test_to_experiment_time_applies_the_offset_and_the_clock_rate():
    assert to_experiment_time(50.0, 20.0) == pytest.approx(70.0)
    assert to_experiment_time(123.0, 4.5) == pytest.approx(127.5)
    # A clock 100 ppm slow has fallen 100 ms behind after a thousand seconds.
    assert to_experiment_time(1000.0, 20.0, 1.0001) == pytest.approx(1020.1)


def test_spectral_features_describe_each_frame_with_several_numbers(data_dir):
    values = spectral_features(data_dir / "three-people-conversation.opus")

    assert values.ndim == 2
    assert values.shape[1] == SPECTRAL_COEFFICIENTS
    # Covering the same 10.4 seconds, at the finer spectral hop.
    assert values.shape[0] * SPECTRAL_HOP == pytest.approx(10.4, rel=0.05)


def test_spectral_features_of_a_silent_video(data_dir):
    assert len(spectral_features(data_dir / "three-people.mp4")) == 0


def test_to_local_time_inverts_to_experiment_time():
    for rate in (1.0, 1.0001, 0.9999):
        for local in (0.0, 50.0, 99.9, 100.0, 150.0, 249.0, 250.0, 400.0):
            experiment = to_experiment_time(local, 20.0, rate)
            assert to_local_time(experiment, 20.0, rate) == pytest.approx(local)


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


def test_offset_curve_follows_a_clock_that_runs_at_a_different_rate():
    room = _room(900)
    other = _drifting_recording(room, 20.0, rate=1.0002)

    points = offset_curve(room, other, offset=20.0, min_quality=TEST_MIN_QUALITY)

    # 200 ppm over the fifteen minutes it covers.
    assert _offset_span(points) == pytest.approx(0.18, abs=0.02)
    assert np.median([p.offset for p in points if p.time < 250]) == pytest.approx(
        20.02, abs=0.03
    )
    assert np.median([p.offset for p in points if p.time > 700]) == pytest.approx(
        20.16, abs=0.03
    )


@pytest.mark.parametrize("rate", [1.0002, 0.9998])
def test_fit_timeline_recovers_the_clock_rate_and_the_offset(rate):
    room = _room(900)
    other = _drifting_recording(room, 20.0, rate=rate)

    fit = fit_timeline(
        offset_curve(room, other, offset=20.0, min_quality=TEST_MIN_QUALITY)
    )

    assert fit is not None
    assert fit.rate == pytest.approx(rate, abs=2e-6)
    assert fit.offset == pytest.approx(20.0, abs=0.02)


def test_fit_timeline_returns_nothing_for_a_steady_recording():
    room = _room(600)
    other = _recording(room, started_later=30.0)

    fit = fit_timeline(
        offset_curve(room, other, offset=30.0, min_quality=TEST_MIN_QUALITY)
    )

    assert fit is None


def test_fit_timeline_ignores_a_difference_too_small_to_trust():
    """Two clocks a part per million apart are not measurably different."""
    points = [
        OffsetPoint(time=t, offset=20.0 + t * 1e-6) for t in np.arange(0.0, 600.0, 5.0)
    ]

    fit = fit_timeline(points, min_drift_ppm=2.0)

    assert fit is None


def test_fit_timeline_needs_a_long_enough_stretch_to_read_a_slope():
    """A slope measured over a few seconds says nothing about a whole session."""
    points = [
        OffsetPoint(time=t, offset=20.0 + t * 1e-4) for t in np.arange(0.0, 20.0, 1.0)
    ]

    fit = fit_timeline(points)

    assert fit is None


def test_fit_timeline_of_nothing_returns_nothing():
    assert fit_timeline([]) is None


def test_fit_timeline_ignores_windows_that_locked_onto_the_wrong_lag():
    """A bad window misses by a second where the rest agree to milliseconds."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 2700.0, 10.0)
    offsets = 20.0 + times * 35e-6 + rng.normal(0.0, 0.005, times.size)
    points = [OffsetPoint(t, o) for t, o in zip(times, offsets)]
    for index in rng.choice(len(points), 20, replace=False):
        points[index] = replace(points[index], offset=points[index].offset + 2.0)

    fit = fit_timeline(points)

    assert fit is not None
    assert fit.drift_ppm == pytest.approx(35.0, abs=1.0)
    assert fit.offset == pytest.approx(20.0, abs=0.01)


def test_fit_timeline_leaves_a_difference_it_cannot_be_sure_of():
    """Half a part per million under five milliseconds of scatter is nothing."""
    rng = np.random.default_rng(1)
    times = np.arange(0.0, 600.0, 10.0)
    points = [
        OffsetPoint(t, 20.0 + t * 5e-7 + n)
        for t, n in zip(times, rng.normal(0.0, 0.005, times.size))
    ]

    fit = fit_timeline(points, min_drift_ppm=0.0)

    assert fit is None


def test_fit_timeline_needs_enough_measurements_to_read_a_slope():
    points = [OffsetPoint(t, 20.0 + t * 1e-4) for t in np.arange(0.0, 700.0, 100.0)]

    fit = fit_timeline(points)

    assert fit is None
