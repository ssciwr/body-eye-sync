from pathlib import Path

import pytest

from body_eye_sync.experiment import preprocess as preprocess_module
from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.timeline import Shift, Timeline
from body_eye_sync.experiment.preprocess import (
    align_experiment,
    apply_timing_corrections,
    clear_timing_corrections,
    has_timing_corrections,
    recordings,
)
from body_eye_sync.preprocessing.alignment import Alignment
from body_eye_sync.preprocessing.timing_correction import (
    FittedTimeline,
    TimingCorrectionAnalysis,
)


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


def test_apply_timing_corrections_only_touches_inputs_that_need_it(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["cam1"].timeline.offset = 0.25
    inputs["mic1"].timeline.offset = 0.5

    corrected = apply_timing_corrections(
        experiment,
        _analysis(
            {
                # Held time: an offset alone is not a correction.
                "cam1": FittedTimeline(Timeline(offset=9.0)),
                "mic1": FittedTimeline(
                    Timeline(
                        offset=0.75,
                        shifts=[Shift(3.0, 0.1)],
                    )
                ),
            }
        ),
    )

    assert corrected == ["mic1"]
    # cam1 keeps the offset alignment gave it, rather than the fit's refinement.
    assert inputs["cam1"].timeline.offset == 0.25
    assert inputs["mic1"].timeline.offset == 0.75
    assert [(s.at, s.seconds) for s in inputs["mic1"].timeline.shifts] == [(3.0, 0.1)]


def test_apply_timing_corrections_ignores_fits_for_absent_inputs(tmp_path):
    experiment = _experiment(tmp_path)

    corrected = apply_timing_corrections(
        experiment,
        _analysis({"gone": FittedTimeline(Timeline(offset=1.0))}),
    )

    assert corrected == []


def test_has_timing_corrections_ignores_offsets(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline.offset = 1.0

    # An offset is where a recording starts, not a correction to its clock.
    assert not has_timing_corrections(experiment)

    inputs["mic1"].timeline.shifts = [Shift(at=5.0, seconds=0.02)]

    assert has_timing_corrections(experiment)


def test_clear_timing_corrections_resets_gaps_only(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline.offset = 1.0
    inputs["mic1"].timeline.shifts = [Shift(at=5.0, seconds=0.02)]

    # Only the input that carried a correction is reported as changed.
    assert clear_timing_corrections(experiment) == ["mic1"]

    assert inputs["mic1"].timeline.shifts == []
    assert inputs["mic1"].timeline.offset == 1.0
    assert not has_timing_corrections(experiment)


def test_clear_timing_corrections_is_a_no_op_when_there_are_none(tmp_path):
    assert clear_timing_corrections(_experiment(tmp_path)) == []


def test_timeline_reports_whether_it_corrects_timing(tmp_path):
    experiment = _experiment(tmp_path)
    inputs = recordings(experiment)
    inputs["mic1"].timeline.offset = 1.0
    inputs["mic1"].timeline.shifts = [Shift(at=5.0, seconds=0.02)]

    assert not inputs["cam1"].timeline.corrects_timing
    assert inputs["mic1"].timeline.corrects_timing
