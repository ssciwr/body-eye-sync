from pathlib import Path

import numpy as np
import pytest

from body_eye_sync.experiment.timeline import Shift, Timeline
from body_eye_sync.preprocessing.timing_correction import (
    OffsetPoint,
    FittedTimeline,
    analyse_timing_corrections,
)


def test_analysis_chooses_the_recording_with_most_overlap_and_returns_global_fits(
    monkeypatch,
):
    features = {
        "long": np.zeros((10_000, 1)),
        "middle": np.zeros((8_000, 1)),
        "short": np.zeros((3_000, 1)),
    }
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.spectral_features",
        lambda path: features[Path(path).stem],
    )
    calls = []

    def measure(reference, other, offset, **kwargs):
        calls.append((reference, other, offset))
        return [OffsetPoint(time=50.0, offset=offset + 0.1)]

    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.offset_curve", measure
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.fit_timeline",
        lambda points, **kwargs: FittedTimeline(
            Timeline(
                offset=points[0].offset,
                shifts=[Shift(at=25.0, seconds=0.2)],
            ),
            residual=0.01,
        ),
    )

    result = analyse_timing_corrections(
        {name: f"{name}.wav" for name in features},
        {"long": 10.0, "middle": 20.0, "short": 70.0},
    )

    assert result.reference == "long"
    assert [call[2] for call in calls] == [10.0, 60.0]
    assert result.fits["long"] == FittedTimeline(Timeline(offset=10.0))
    assert result.fits["middle"].timeline.offset == pytest.approx(20.1)
    assert result.points["middle"][0].time == pytest.approx(60.0)
    assert result.points["middle"][0].offset == pytest.approx(20.1)


def test_analysis_reports_inputs_without_usable_audio(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.spectral_features",
        lambda path: (
            np.zeros((1_000, 1)) if Path(path).stem != "silent" else np.empty((0, 1))
        ),
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.offset_curve",
        lambda *args, **kwargs: [OffsetPoint(5.0, 0.0)],
    )

    result = analyse_timing_corrections(
        {"one": "one.wav", "two": "two.wav", "silent": "silent.wav"},
        {"one": 0.0, "two": 0.0, "silent": 0.0},
    )

    assert result.unavailable == ["silent"]


def test_analysis_needs_two_audio_inputs(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.spectral_features",
        lambda path: (
            np.zeros((100, 1)) if Path(path).stem == "one" else np.empty((0, 1))
        ),
    )

    with pytest.raises(ValueError, match="fewer than two"):
        analyse_timing_corrections(
            {"one": "one.wav", "silent": "silent.wav"},
            {"one": 0.0, "silent": 0.0},
        )


@pytest.mark.parametrize(
    ("shifts", "expected_shift_count"),
    [
        ([Shift(5.0, 0.05), Shift(10.0, 0.06)], 2),
        ([Shift(10.0, 0.09)], 0),
    ],
)
def test_analysis_suppresses_total_gaps_below_the_sensitivity_floor(
    monkeypatch, shifts, expected_shift_count
):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.spectral_features",
        lambda path: np.zeros((1_000, 1)),
    )
    points = [
        OffsetPoint(20.0, 1.0),
        OffsetPoint(40.0, 1.01),
    ]
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.offset_curve",
        lambda *args, **kwargs: points,
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.fit_timeline",
        lambda *args, **kwargs: FittedTimeline(Timeline(offset=1.0, shifts=shifts)),
    )

    result = analyse_timing_corrections(
        {"reference": "reference.wav", "other": "other.wav"},
        {"reference": 0.0, "other": 1.0},
    )

    assert len(result.fits["other"].timeline.shifts) == expected_shift_count


def test_analysis_forwards_the_measurement_settings(monkeypatch):
    features = {"a": np.zeros((5_000, 1)), "b": np.zeros((5_000, 1))}
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.spectral_features",
        lambda path: features[Path(path).stem],
    )
    seen = {}

    def curve(reference, other, offset, **kwargs):
        seen.update(kwargs)
        return [OffsetPoint(10.0, 0.0)]

    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.offset_curve", curve
    )

    analyse_timing_corrections(
        {"a": Path("a.wav"), "b": Path("b.wav")},
        {"a": 0.0, "b": 0.0},
        window=20.0,
        search=30.0,
        min_quality=4.5,
    )

    assert seen["window"] == 20.0
    assert seen["search"] == 30.0
    assert seen["min_quality"] == 4.5


def test_min_shift_reaches_the_fit(monkeypatch):
    features = {"a": np.zeros((5_000, 1)), "b": np.zeros((5_000, 1))}
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.spectral_features",
        lambda path: features[Path(path).stem],
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.offset_curve",
        lambda *args, **kwargs: [OffsetPoint(10.0, 0.0)],
    )
    seen = {}

    def fit(points, **kwargs):
        seen.update(kwargs)
        return FittedTimeline(Timeline())

    monkeypatch.setattr(
        "body_eye_sync.preprocessing.timing_correction.fit_timeline", fit
    )

    analyse_timing_corrections(
        {"a": Path("a.wav"), "b": Path("b.wav")},
        {"a": 0.0, "b": 0.0},
        min_shift=0.04,
    )

    assert seen["min_shift"] == 0.04
