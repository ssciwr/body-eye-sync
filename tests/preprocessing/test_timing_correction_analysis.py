from pathlib import Path

import numpy as np
import pytest

from body_eye_sync.preprocessing.alignment import (
    DriftPoint,
    Method,
    Shift,
    TimelineFit,
)
from body_eye_sync.preprocessing.timing_correction import (
    analyse_timing_corrections,
)


def spectral_method(extract):
    return Method("spectral", extract, 0.01, 7.0)


def test_analysis_chooses_the_recording_with_most_overlap_and_returns_global_fits(
    monkeypatch,
):
    features = {
        "long": np.zeros((10_000, 1)),
        "middle": np.zeros((8_000, 1)),
        "short": np.zeros((3_000, 1)),
    }
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.SPECTRAL",
        spectral_method(lambda path: features[Path(path).stem]),
    )
    calls = []

    def measure(reference, other, offset, **kwargs):
        calls.append((reference, other, offset))
        return [DriftPoint(time=50.0, offset=offset + 0.1, quality=20.0)]

    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.drift_curve", measure
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.fit_timeline",
        lambda points, **kwargs: TimelineFit(
            offset=points[0].offset,
            scale=1.0001,
            shifts=[Shift(at=25.0, seconds=0.2)],
            residual=0.01,
        ),
    )

    result = analyse_timing_corrections(
        {name: f"{name}.wav" for name in features},
        {"long": 10.0, "middle": 20.0, "short": 70.0},
    )

    assert result.reference == "long"
    assert [call[2] for call in calls] == [10.0, 60.0]
    assert result.fits["long"] == TimelineFit(offset=10.0, scale=1.0)
    assert result.fits["middle"].offset == pytest.approx(20.1)
    assert result.fits["middle"].scale == pytest.approx(1.0001)
    assert result.points["middle"][0].time == pytest.approx(60.0)
    assert result.points["middle"][0].offset == pytest.approx(20.1)


def test_analysis_reports_inputs_without_usable_audio(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.SPECTRAL",
        spectral_method(
            lambda path: (
                np.zeros((1_000, 1))
                if Path(path).stem != "silent"
                else np.empty((0, 1))
            )
        ),
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.drift_curve",
        lambda *args, **kwargs: [DriftPoint(5.0, 0.0, 20.0)],
    )

    result = analyse_timing_corrections(
        {"one": "one.wav", "two": "two.wav", "silent": "silent.wav"},
        {"one": 0.0, "two": 0.0, "silent": 0.0},
    )

    assert result.unavailable == ["silent"]


def test_analysis_needs_two_audio_inputs(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.SPECTRAL",
        spectral_method(
            lambda path: (
                np.zeros((100, 1)) if Path(path).stem == "one" else np.empty((0, 1))
            )
        ),
    )

    with pytest.raises(ValueError, match="fewer than two"):
        analyse_timing_corrections(
            {"one": "one.wav", "silent": "silent.wav"},
            {"one": 0.0, "silent": 0.0},
        )


@pytest.mark.parametrize(
    ("scale", "shifts", "expected_scale", "expected_shift_count"),
    [
        (1.000020, [Shift(5.0, 0.05), Shift(10.0, 0.06)], 1.0, 2),
        (1.000100, [Shift(10.0, 0.09)], 1.000100, 0),
        (1.000020, [Shift(10.0, 0.09)], 1.0, 0),
    ],
)
def test_analysis_suppresses_drift_and_total_gaps_below_the_sensitivity_floor(
    monkeypatch, scale, shifts, expected_scale, expected_shift_count
):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.SPECTRAL",
        spectral_method(lambda path: np.zeros((1_000, 1))),
    )
    points = [
        DriftPoint(20.0, 1.0, 20.0),
        DriftPoint(40.0, 1.01, 20.0),
    ]
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.drift_curve",
        lambda *args, **kwargs: points,
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.fit_timeline",
        lambda *args, **kwargs: TimelineFit(1.0, scale, shifts),
    )

    result = analyse_timing_corrections(
        {"reference": "reference.wav", "other": "other.wav"},
        {"reference": 0.0, "other": 1.0},
    )

    assert result.fits["other"].scale == pytest.approx(expected_scale)
    assert len(result.fits["other"].shifts) == expected_shift_count
