import pytest

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.timing_correction import TimingCorrectionTab
from body_eye_sync.preprocessing.alignment import DriftPoint, Shift, TimelineFit
from body_eye_sync.preprocessing.timing_correction import TimingCorrectionAnalysis


@pytest.fixture
def experiment(data_dir):
    return Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
            ],
            glasses_videos=[
                GlassesVideoInput(
                    id="glasses",
                    path=data_dir / "three-people.mp4",
                    gaze_path=data_dir / "three-people.tsv",
                    time_offset=5.0,
                )
            ],
        )
    )


@pytest.fixture
def tab(qtbot, experiment):
    widget = TimingCorrectionTab(experiment)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def analysis():
    return TimingCorrectionAnalysis(
        reference="room",
        points={
            "room": [],
            "glasses": [
                DriftPoint(60.0, 5.2, 20.0),
                DriftPoint(120.0, 5.3, 20.0),
            ],
        },
        fits={
            "room": TimelineFit(offset=0.0, scale=1.0),
            "glasses": TimelineFit(
                offset=5.2,
                scale=1.0001,
                shifts=[
                    Shift(at=25.3, seconds=0.018),
                    Shift(at=99.1, seconds=0.062),
                ],
                residual=0.01,
            ),
        },
        unavailable=[],
    )


def _row(tab, input_id):
    return next(
        row
        for row in range(tab.table.rowCount())
        if tab.table.item(row, 0).text() == input_id
    )


def test_scrollable_summary_table_shows_every_input(tab):
    headers = [
        tab.table.horizontalHeaderItem(column).text()
        for column in range(tab.table.columnCount())
    ]

    assert headers == ["Id", "Offset", "Drift", "Gaps"]
    assert tab.table.rowCount() == 2
    assert tab.scroll_area.widget() is tab.page
    assert tab.scroll_area.widgetResizable()
    assert tab.page.layout().itemAt(0).widget() is tab.correct_button
    assert tab.page.layout().itemAt(1).widget() is tab.table
    glasses = _row(tab, "glasses")
    assert [tab.table.item(glasses, column).text() for column in range(4)] == [
        "glasses",
        "+5.000 s",
        "0 ppm",
        "0",
    ]
    assert tab.correct_button.text() == "Analyse and correct timing"


def test_existing_corrections_are_plotted_as_lines_without_data_points(tab):
    glasses = tab.experiment.glasses_videos[0]
    glasses.time_scale = 1.0001
    glasses.time_shifts = [Shift(at=0.05, seconds=0.12)]

    tab.refresh()

    assert tab.canvas.isVisibleTo(tab)
    assert tab.figure.axes[0].lines
    assert len(tab.figure.axes[0].collections) == 0


def test_correction_recalculates_applies_and_plots_lines(
    qtbot, tab, analysis, monkeypatch
):
    measured_offsets = []

    def calculate(paths, offsets, progress):
        measured_offsets.append(offsets)
        return analysis

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.timing_correction.analyse_timing_corrections",
        calculate,
    )

    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    glasses = tab.experiment.glasses_videos[0]
    assert measured_offsets == [{"glasses": 5.0, "room": 0.0}]
    assert glasses.time_offset == pytest.approx(5.2)
    assert glasses.time_scale == pytest.approx(1.0001)
    assert [(shift.at, shift.seconds) for shift in glasses.time_shifts] == [
        (25.3, 0.018),
        (99.1, 0.062),
    ]
    row = _row(tab, "glasses")
    assert [tab.table.item(row, column).text() for column in range(1, 4)] == [
        "+5.200 s",
        "+100 ppm",
        "25.3s: 18ms, 99.1s: 62ms",
    ]
    assert tab.canvas.isVisibleTo(tab)
    assert tab.figure.axes[0].lines
    assert tab.figure.axes[0].collections


def test_an_input_that_held_time_keeps_the_offset_alignment_gave_it(
    qtbot, tab, monkeypatch
):
    """Correcting one input does not quietly re-place the others.

    ``room`` here comes back measured but with nothing to correct, and its fitted
    offset differs from the one it already has. Refining it would be a change the
    user did not ask this tab for, so it is left alone.
    """
    analysis = TimingCorrectionAnalysis(
        reference="glasses",
        points={"glasses": [], "room": [DriftPoint(60.0, 0.4, 20.0)]},
        fits={
            "glasses": TimelineFit(
                offset=5.2, scale=1.0001, shifts=[Shift(at=25.3, seconds=0.018)]
            ),
            # Measured, no drift and no gaps, but not where it currently sits.
            "room": TimelineFit(offset=0.4, scale=1.0),
        },
        unavailable=[],
    )
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.timing_correction.analyse_timing_corrections",
        lambda paths, offsets, progress: analysis,
    )

    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    room = tab.experiment.fixed_videos[0]
    assert room.time_offset == 0.0
    assert room.time_scale == 1.0
    assert room.time_shifts == []
    # The input that did need correcting still got one.
    assert tab.experiment.glasses_videos[0].time_offset == pytest.approx(5.2)
    assert tab.summary_label.text() == "Corrected drift or gaps in 1 input(s)."


def test_no_detectable_correction_shows_only_the_message(qtbot, tab, monkeypatch):
    analysis = TimingCorrectionAnalysis(
        reference="room",
        points={"room": [], "glasses": [DriftPoint(60.0, 4.75, 20.0)]},
        fits={
            "room": TimelineFit(offset=0.0, scale=1.0),
            "glasses": TimelineFit(offset=4.75, scale=1.0),
        },
        unavailable=[],
    )
    changed = []
    tab.experiment_changed.connect(lambda: changed.append(True))
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.timing_correction.analyse_timing_corrections",
        lambda paths, offsets, progress: analysis,
    )

    tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    assert tab.summary_label.text() == "No drift or gaps detected"
    assert tab.canvas is None or not tab.canvas.isVisibleTo(tab)
    assert tab.experiment.glasses_videos[0].time_offset == 5.0
    assert tab.experiment.glasses_videos[0].time_scale == 1.0
    assert tab.experiment.glasses_videos[0].time_shifts == []
    assert changed == []


def test_correction_requires_two_inputs(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
            ]
        )
    )
    tab = TimingCorrectionTab(experiment)
    qtbot.addWidget(tab)

    assert not tab.correct_button.isEnabled()
