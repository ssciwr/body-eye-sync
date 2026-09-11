from pathlib import Path

import numpy as np
import pytest

from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.preprocessing.clock_rate import (
    OffsetPoint,
    analyse_clock_rates,
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
        "body_eye_sync.preprocessing.clock_rate.spectral_features",
        lambda path: features[Path(path).stem],
    )
    calls = []

    def measure(reference, other, offset, **kwargs):
        calls.append((reference, other, offset))
        return [OffsetPoint(time=50.0, offset=offset + 0.1)]

    monkeypatch.setattr("body_eye_sync.preprocessing.clock_rate.offset_curve", measure)
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.fit_timeline",
        lambda points, **kwargs: Timeline(offset=points[0].offset, rate=1.00002),
    )

    result = analyse_clock_rates(
        {name: f"{name}.wav" for name in features},
        {"long": 10.0, "middle": 20.0, "short": 70.0},
    )

    assert result.reference == "long"
    assert [call[2] for call in calls] == [10.0, 60.0]
    assert "long" not in result.fits
    assert result.fits["middle"].offset == pytest.approx(20.1)
    assert result.points["middle"][0].time == pytest.approx(60.0)
    assert result.points["middle"][0].offset == pytest.approx(20.1)


def test_analysis_reports_inputs_without_usable_audio(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.spectral_features",
        lambda path: (
            np.zeros((1_000, 1)) if Path(path).stem != "silent" else np.empty((0, 1))
        ),
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.offset_curve",
        lambda *args, **kwargs: [OffsetPoint(5.0, 0.0)],
    )

    result = analyse_clock_rates(
        {"one": "one.wav", "two": "two.wav", "silent": "silent.wav"},
        {"one": 0.0, "two": 0.0, "silent": 0.0},
    )

    assert result.unavailable == ["silent"]


def test_analysis_needs_two_audio_inputs(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.spectral_features",
        lambda path: (
            np.zeros((100, 1)) if Path(path).stem == "one" else np.empty((0, 1))
        ),
    )

    with pytest.raises(ValueError, match="fewer than two"):
        analyse_clock_rates(
            {"one": "one.wav", "silent": "silent.wav"},
            {"one": 0.0, "silent": 0.0},
        )


def test_analysis_forwards_the_measurement_settings(monkeypatch):
    features = {"a": np.zeros((5_000, 1)), "b": np.zeros((5_000, 1))}
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.spectral_features",
        lambda path: features[Path(path).stem],
    )
    seen = {}

    def curve(reference, other, offset, **kwargs):
        seen.update(kwargs)
        return [OffsetPoint(10.0, 0.0)]

    monkeypatch.setattr("body_eye_sync.preprocessing.clock_rate.offset_curve", curve)

    analyse_clock_rates(
        {"a": Path("a.wav"), "b": Path("b.wav")},
        {"a": 0.0, "b": 0.0},
        window=20.0,
        search=30.0,
        min_quality=4.5,
    )

    assert seen["window"] == 20.0
    assert seen["search"] == 30.0
    assert seen["min_quality"] == 4.5


def test_min_drift_reaches_the_fit(monkeypatch):
    features = {"a": np.zeros((5_000, 1)), "b": np.zeros((5_000, 1))}
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.spectral_features",
        lambda path: features[Path(path).stem],
    )
    monkeypatch.setattr(
        "body_eye_sync.preprocessing.clock_rate.offset_curve",
        lambda *args, **kwargs: [OffsetPoint(10.0, 0.0)],
    )
    seen = {}

    def fit(points, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr("body_eye_sync.preprocessing.clock_rate.fit_timeline", fit)

    analyse_clock_rates(
        {"a": Path("a.wav"), "b": Path("b.wav")},
        {"a": 0.0, "b": 0.0},
        min_drift_ppm=4.0,
    )

    assert seen["min_drift_ppm"] == 4.0
