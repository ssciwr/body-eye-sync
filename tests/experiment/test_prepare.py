from pathlib import Path

import pytest

from body_eye_sync.experiment import prepare as prepare_module
from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.prepare import (
    align_experiment,
    apply_timing_corrections,
    correct_experiment_timing,
    prepare_experiment,
    recordings,
    stored_fits,
)
from body_eye_sync.preprocessing.alignment import Alignment, Shift, TimelineFit
from body_eye_sync.preprocessing.timing_correction import TimingCorrectionAnalysis


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


def _analysis(fits, unavailable=()):
    return TimingCorrectionAnalysis(
        reference="cam1", points={}, fits=fits, unavailable=list(unavailable)
    )


def test_recordings_keys_inputs_by_id(tmp_path):
    experiment = _experiment(tmp_path)

    assert list(recordings(experiment)) == ["cam1", "mic1"]


def test_align_experiment_writes_the_offsets_onto_the_inputs(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    monkeypatch.setattr(
        prepare_module,
        "align_media",
        lambda paths, **kwargs: Alignment(offsets={"cam1": 0.0, "mic1": 1.5}),
    )

    result = align_experiment(experiment)

    assert result.offsets == {"cam1": 0.0, "mic1": 1.5}
    assert recordings(experiment)["mic1"].time_offset == 1.5


def test_align_experiment_needs_two_recordings(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path, ids=("cam1",))
    monkeypatch.setattr(
        prepare_module,
        "align_media",
        lambda paths, **kwargs: pytest.fail("should not measure a lone recording"),
    )

    assert align_experiment(experiment).offsets == {}


def test_apply_timing_corrections_only_touches_inputs_that_need_it(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["cam1"].time_offset = 0.25
    inputs["mic1"].time_offset = 0.5

    corrected = apply_timing_corrections(
        experiment,
        _analysis(
            {
                # Held time: an offset alone is not a correction.
                "cam1": TimelineFit(offset=9.0, scale=1.0),
                "mic1": TimelineFit(
                    offset=0.75, scale=1.000002, shifts=[Shift(3.0, 0.1)]
                ),
            }
        ),
    )

    assert corrected == ["mic1"]
    # cam1 keeps the offset alignment gave it, rather than the fit's refinement.
    assert inputs["cam1"].time_offset == 0.25
    assert inputs["cam1"].time_scale == 1.0
    # mic1 takes the offset that goes with its slope, as a matched pair.
    assert inputs["mic1"].time_offset == 0.75
    assert inputs["mic1"].time_scale == 1.000002
    assert [(s.at, s.seconds) for s in inputs["mic1"].time_shifts] == [(3.0, 0.1)]


def test_apply_timing_corrections_ignores_fits_for_absent_inputs(tmp_path):
    experiment = _experiment(tmp_path)

    corrected = apply_timing_corrections(
        experiment, _analysis({"gone": TimelineFit(offset=1.0, scale=1.0001)})
    )

    assert corrected == []


def test_correct_experiment_timing_measures_against_the_stored_offsets(
    tmp_path, monkeypatch
):
    experiment = _experiment(tmp_path)
    recordings(experiment)["mic1"].time_offset = 2.0
    seen = {}

    def fake_analyse(paths, offsets, **kwargs):
        seen["offsets"] = offsets
        return _analysis({"mic1": TimelineFit(offset=2.0, scale=1.00001)})

    monkeypatch.setattr(prepare_module, "analyse_timing_corrections", fake_analyse)

    analysis, corrected = correct_experiment_timing(experiment)

    assert seen["offsets"] == {"cam1": 0.0, "mic1": 2.0}
    assert corrected == ["mic1"]
    assert analysis is not None


def test_stored_fits_report_what_the_inputs_already_carry(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].time_offset = 1.0
    inputs["mic1"].time_scale = 1.0001

    fits = stored_fits(experiment)

    assert not fits["cam1"].corrects_timing
    assert fits["mic1"].corrects_timing
    assert fits["mic1"].offset == 1.0


def test_prepare_experiment_aligns_before_correcting(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    order = []

    def fake_align(paths, **kwargs):
        order.append("align")
        return Alignment(offsets={"cam1": 0.0, "mic1": 4.0})

    def fake_analyse(paths, offsets, **kwargs):
        order.append("correct")
        # Drift is measured against the offsets alignment has just written.
        assert offsets["mic1"] == 4.0
        return _analysis({"mic1": TimelineFit(offset=4.0, scale=1.00003)})

    monkeypatch.setattr(prepare_module, "align_media", fake_align)
    monkeypatch.setattr(prepare_module, "analyse_timing_corrections", fake_analyse)

    analysis = prepare_experiment(experiment)

    assert order == ["align", "correct"]
    assert analysis is not None
    assert recordings(experiment)["mic1"].time_scale == 1.00003


def test_prepare_experiment_reports_progress_over_both_stages(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)

    def fake_align(paths, progress=None):
        progress(0.0)
        progress(1.0)
        return Alignment(offsets={})

    def fake_analyse(paths, offsets, progress=None):
        progress(0.0)
        progress(1.0)
        return _analysis({})

    monkeypatch.setattr(prepare_module, "align_media", fake_align)
    monkeypatch.setattr(prepare_module, "analyse_timing_corrections", fake_analyse)

    seen = []
    prepare_experiment(experiment, progress=lambda value: seen.append(value) or True)

    # Each stage gets half the bar, in order, so it only ever moves forwards.
    assert seen == [0.0, 0.5, 0.5, 1.0]


def test_prepare_experiment_does_nothing_with_one_recording(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path, ids=("cam1",))
    monkeypatch.setattr(
        prepare_module,
        "align_media",
        lambda paths, **kwargs: pytest.fail("should not measure a lone recording"),
    )

    assert prepare_experiment(experiment) is None
