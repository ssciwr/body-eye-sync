import numpy as np
import pytest

from body_eye_sync.glasses import Recording, TrackingData


def _data(count=5, resolution=(1920, 1080), **overrides):
    """Tracking data whose gaze walks across the frame, one sample untracked."""
    gaze = np.column_stack(
        [np.linspace(0.1, 0.5, count), np.full(count, 0.5)],
    )
    if count > 2:
        gaze[2] = np.nan
    fields = dict(
        time=np.arange(count) * 0.02,
        gaze=gaze,
        pupil=np.full((count, 2), 3.5),
        recording=Recording(source="somewhere", device="test", resolution=resolution),
    )
    return TrackingData(**{**fields, **overrides})


def test_length_and_duration_and_rate():
    data = _data(count=5)

    assert len(data) == 5
    assert data.duration == pytest.approx(0.08)
    assert data.sample_rate == pytest.approx(50.0)


def test_a_single_sample_has_no_duration_or_rate():
    data = _data(count=1)

    assert data.duration == 0.0
    assert data.sample_rate == 0.0


def test_the_rate_is_the_one_it_ran_at_not_the_one_it_managed():
    """A device that drops samples was still running at its own rate."""
    data = _data(count=4, time=np.array([0.0, 0.02, 0.28, 0.30]))

    assert data.sample_rate == pytest.approx(50.0)


def test_valid_marks_the_samples_with_a_gaze_point():
    assert list(_data().valid) == [True, True, False, True, True]


def test_gaze_in_pixels_uses_the_recordings_own_frame_size():
    pixels = _data().gaze_pixels()

    assert pixels[0] == pytest.approx([0.1 * 1920, 0.5 * 1080])
    assert np.isnan(pixels[2]).all()


def test_gaze_in_pixels_can_be_asked_for_another_frame_size():
    assert _data().gaze_pixels((100, 200))[0] == pytest.approx([10.0, 100.0])


def test_gaze_in_pixels_needs_a_frame_size_from_somewhere():
    data = _data(recording=None)

    with pytest.raises(ValueError, match="frame size"):
        data.gaze_pixels()


def test_between_takes_the_samples_of_a_span():
    window = _data().between(0.02, 0.06)

    assert window.time == pytest.approx([0.02, 0.04])
    assert len(window.pupil) == 2
    assert window.recording.resolution == (1920, 1080)


def test_between_keeps_the_optional_arrays_in_step():
    count = 5
    data = _data(
        count=count,
        gaze_3d=np.zeros((count, 3)),
        eye_origin=np.zeros((count, 2, 3)),
        eye_direction=np.zeros((count, 2, 3)),
    )

    window = data.between(0.0, 0.04)

    assert len(window) == 2
    assert window.gaze_3d.shape == (2, 3)
    assert window.eye_origin.shape == (2, 2, 3)
    assert window.eye_direction.shape == (2, 2, 3)


def test_mean_gaze_averages_a_frames_worth_of_samples():
    data = _data()

    assert data.mean_gaze(0.0, 0.04) == pytest.approx([0.15, 0.5])


def test_mean_gaze_ignores_samples_that_tracked_nothing():
    data = _data()

    # 0.04 was not tracked, so only 0.06 counts towards the frame.
    assert data.mean_gaze(0.04, 0.08) == pytest.approx([0.4, 0.5])


def test_mean_gaze_of_a_span_that_tracked_nothing_at_all():
    data = _data()

    assert np.isnan(data.mean_gaze(0.04, 0.06)).all()
    assert np.isnan(data.mean_gaze(9.0, 10.0)).all()


def test_arrays_that_do_not_match_the_samples_are_refused():
    with pytest.raises(ValueError, match=r"pupil has shape"):
        TrackingData(time=np.zeros(3), gaze=np.zeros((3, 2)), pupil=np.zeros((2, 2)))


def _series(count=200, rate=100.0, value=(0.0, 0.0, 0.0)):
    from body_eye_sync.glasses import Series

    return Series(
        np.arange(count) / rate, np.tile(np.asarray(value, float), (count, 1))
    )


def test_a_series_measures_its_own_rate():
    assert _series(rate=120.0).sample_rate == pytest.approx(120.0)
    assert _series(count=1).sample_rate == 0.0


def test_a_series_takes_the_readings_of_a_span():
    window = _series(rate=100.0).between(0.10, 0.15)

    assert len(window) == 5
    assert window.time == pytest.approx([0.10, 0.11, 0.12, 0.13, 0.14])


def test_a_series_refuses_values_that_do_not_match_its_times():
    from body_eye_sync.glasses import Series

    with pytest.raises(ValueError, match="expected"):
        Series(np.zeros(3), np.zeros((2, 3)))


def test_the_gyroscope_bias_comes_from_the_moments_the_head_was_still():
    """A still head reads its own bias, not zero, so stillness is steadiness."""
    from body_eye_sync.glasses import MotionData, Series

    bias = np.array([0.4, 2.3, -2.6])
    rate, seconds = 100, 6
    time = np.arange(rate * seconds) / rate
    values = np.tile(bias, (len(time), 1))
    # The middle two seconds are a head turn: a rate that rises and falls,
    # since a rate that merely sat at some constant would look just as steady
    # as a still head does.
    turning = np.sin(np.linspace(0, np.pi, 2 * rate))[:, None]
    values[2 * rate : 4 * rate] += turning * np.array([40.0, -30.0, 10.0])
    motion = MotionData(
        acceleration=Series.empty(), angular_velocity=Series(time, values)
    )

    assert motion.angular_velocity_bias() == pytest.approx(bias)


def test_the_gyroscope_bias_of_a_recording_that_was_never_still():
    from body_eye_sync.glasses import MotionData, Series

    time = np.arange(500) / 100
    moving = (
        np.column_stack([np.sin(time * 9), np.cos(time * 7), np.sin(time * 5)]) * 50
    )
    motion = MotionData(
        acceleration=Series.empty(), angular_velocity=Series(time, moving)
    )

    assert np.isnan(motion.angular_velocity_bias()).all()


def test_motion_data_of_a_source_that_recorded_none():
    from body_eye_sync.glasses import MotionData

    motion = MotionData.empty()

    assert len(motion) == 0
    assert np.isnan(motion.angular_velocity_bias()).all()
