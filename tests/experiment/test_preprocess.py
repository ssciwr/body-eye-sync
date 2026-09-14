from pathlib import Path

import pytest

from body_eye_sync.experiment import preprocess as preprocess_module
from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.experiment.preprocess import (
    align_experiment,
    apply_clock_rates,
    clear_clock_rates,
    has_corrected_clock_rates,
    recordings,
)
from body_eye_sync.preprocessing.alignment import Alignment
from body_eye_sync.preprocessing.clock_rate import ClockRateAnalysis


def _experiment(tmp_path, ids=("cam1", "mic1")):
    config = ExperimentConfig(
        glasses_videos=[
            GlassesVideoInput(
                id=ids[0],
                path=Path("videos/example.mp4"),
                gaze_path=Path("videos/example.tsv"),
            )
        ],
        audio=[AudioInput(id=name, path=Path(f"audio/{name}.wav")) for name in ids[1:]],
    )
    return Experiment(config, tmp_path)


def _analysis(fits, points=None, unavailable=()):
    return ClockRateAnalysis(
        reference="cam1",
        points={} if points is None else points,
        fits=fits,
        unavailable=list(unavailable),
    )


def test_recordings_keys_inputs_by_id(tmp_path):
    experiment = _experiment(tmp_path)

    assert list(recordings(experiment)) == ["cam1", "mic1"]


def test_align_experiment_writes_the_offsets_onto_the_inputs(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    monkeypatch.setattr(
        preprocess_module,
        "align_media",
        lambda paths, **kwargs: Alignment(offsets={"cam1": 0.0, "mic1": 1.5}),
    )

    result = align_experiment(experiment)

    assert result.offsets == {"cam1": 0.0, "mic1": 1.5}
    assert recordings(experiment)["mic1"].timeline.offset == 1.5


def test_align_experiment_needs_two_recordings(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path, ids=("cam1",))
    monkeypatch.setattr(
        preprocess_module,
        "align_media",
        lambda paths, **kwargs: pytest.fail("should not measure a lone recording"),
    )

    assert align_experiment(experiment).offsets == {}


def test_apply_clock_rates_applies_only_significant_drift_fits(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["cam1"].timeline.offset = 0.25
    inputs["mic1"].timeline.offset = 0.5

    corrected = apply_clock_rates(
        experiment,
        _analysis(
            {"mic1": Timeline(offset=0.75, rate=1.00003)},
            points={"cam1": [], "mic1": []},
        ),
    )

    assert corrected == ["mic1"]
    assert inputs["cam1"].timeline.offset == 0.25
    assert inputs["mic1"].timeline.offset == 0.75
    assert inputs["mic1"].timeline.rate == pytest.approx(1.00003)


def test_apply_clock_rates_can_replace_an_old_rate_with_one(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline = Timeline(offset=0.5, rate=1.00003)

    changed = apply_clock_rates(
        experiment,
        _analysis({}, points={"cam1": [], "mic1": []}),
    )

    assert changed == ["mic1"]
    assert inputs["mic1"].timeline == Timeline(offset=0.5, rate=1.0)


def test_apply_clock_rates_ignores_fits_for_absent_inputs(tmp_path):
    experiment = _experiment(tmp_path)

    corrected = apply_clock_rates(
        experiment,
        _analysis({"gone": Timeline(offset=1.0, rate=1.00003)}),
    )

    assert corrected == []


def test_has_corrected_clock_rates_ignores_offsets(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline.offset = 1.0

    # An offset is where a recording starts, not a correction to its clock.
    assert not has_corrected_clock_rates(experiment)

    inputs["mic1"].timeline.rate = 1.00003

    assert has_corrected_clock_rates(experiment)


def test_clear_clock_rates_resets_clock_rates_only(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline.offset = 1.0
    inputs["mic1"].timeline.rate = 1.00003

    # Only the input that carried a correction is reported as changed.
    assert clear_clock_rates(experiment) == ["mic1"]

    assert inputs["mic1"].timeline.rate == 1.0
    assert inputs["mic1"].timeline.offset == 1.0
    assert not has_corrected_clock_rates(experiment)


def test_clear_clock_rates_is_a_no_op_when_there_are_none(tmp_path):
    assert clear_clock_rates(_experiment(tmp_path)) == []


def test_timeline_reports_whether_its_clock_ran_at_a_different_rate(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline.offset = 1.0
    inputs["mic1"].timeline.rate = 1.00003

    assert not inputs["cam1"].timeline.corrects_drift
    assert inputs["mic1"].timeline.corrects_drift
    assert inputs["mic1"].timeline.drift_ppm == pytest.approx(30.0)
