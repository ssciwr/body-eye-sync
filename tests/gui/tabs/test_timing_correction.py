import threading

import pytest

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs.timing_correction import TimingCorrectionTab
from body_eye_sync.preprocessing.alignment import (
    DEFAULT_MIN_SHIFT,
    SPECTRAL,
    DriftPoint,
    Shift,
    TimelineFit,
)
from body_eye_sync.preprocessing.timing_correction import (
    DEFAULT_SEARCH,
    DEFAULT_WINDOW,
    TimingCorrectionAnalysis,
)


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
    # Table on the left, settings and buttons on the right, plot under both.
    top = tab.page.layout().itemAt(0).layout()
    assert top.itemAt(0).widget() is tab.table
    controls = top.itemAt(1).layout()
    assert controls.itemAt(0).widget() is tab.settings_widget
    buttons = controls.itemAt(1).layout()
    assert buttons.itemAt(0).widget() is tab.correct_button
    assert buttons.itemAt(1).widget() is tab.clear_button
    assert tab.page.layout().itemAt(1).widget() is tab.canvas
    glasses = _row(tab, "glasses")
    assert [tab.table.item(glasses, column).text() for column in range(4)] == [
        "glasses",
        "+5.000 s",
        "0 ppm",
        "0",
    ]
    assert tab.correct_button.text() == "Analyse and correct timing"
    assert tab.clear_button.text() == "Clear corrections"


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

    def calculate(paths, offsets, progress, **settings):
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
        lambda paths, offsets, progress, **settings: analysis,
    )
    messages = []
    tab.status_message.connect(messages.append)

    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    room = tab.experiment.fixed_videos[0]
    assert room.time_offset == 0.0
    assert room.time_scale == 1.0
    assert room.time_shifts == []
    # The input that did need correcting still got one.
    assert tab.experiment.glasses_videos[0].time_offset == pytest.approx(5.2)
    assert messages == ["Corrected drift or gaps in 1 input(s)"]


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
        lambda paths, offsets, progress, **settings: analysis,
    )
    messages = []
    tab.status_message.connect(messages.append)

    tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    assert messages == ["No drift or gaps detected"]
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


def test_clear_is_disabled_until_there_is_something_to_clear(tab):
    assert not tab.clear_button.isEnabled()

    tab.experiment.glasses_videos[0].time_scale = 1.0001
    tab.refresh()

    assert tab.clear_button.isEnabled()


def test_clear_resets_drift_and_gaps_but_keeps_the_offsets(qtbot, tab):
    glasses = tab.experiment.glasses_videos[0]
    glasses.time_scale = 1.0001
    glasses.time_shifts = [Shift(at=25.3, seconds=0.018)]
    tab.refresh()
    messages = []
    tab.status_message.connect(messages.append)

    with qtbot.waitSignal(tab.experiment_changed):
        tab.clear_button.click()

    assert glasses.time_scale == 1.0
    assert glasses.time_shifts == []
    # Where the recording starts is alignment's answer, not this tab's to undo.
    assert glasses.time_offset == 5.0
    row = _row(tab, "glasses")
    assert [tab.table.item(row, column).text() for column in range(1, 4)] == [
        "+5.000 s",
        "0 ppm",
        "0",
    ]
    assert messages == ["Timing corrections cleared"]
    assert not tab.canvas.isVisibleTo(tab)
    assert not tab.clear_button.isEnabled()


def test_clear_drops_a_fresh_analysis_plot(qtbot, tab, analysis, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.timing_correction.analyse_timing_corrections",
        lambda paths, offsets, progress, **settings: analysis,
    )
    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())
    assert tab.canvas.isVisibleTo(tab)

    with qtbot.waitSignal(tab.experiment_changed):
        tab.clear_button.click()

    assert tab.experiment.glasses_videos[0].time_scale == 1.0
    assert tab.experiment.glasses_videos[0].time_shifts == []
    # The corrected offset stays: clearing does not re-run alignment.
    assert tab.experiment.glasses_videos[0].time_offset == pytest.approx(5.2)
    assert not tab.canvas.isVisibleTo(tab)


def test_settings_default_to_the_analysis_defaults(tab):
    assert tab.window_spin.value() == pytest.approx(DEFAULT_WINDOW)
    assert tab.search_spin.value() == pytest.approx(DEFAULT_SEARCH)
    assert tab.min_quality_spin.value() == pytest.approx(SPECTRAL.min_quality)
    assert tab.min_gap_spin.value() == pytest.approx(1000 * DEFAULT_MIN_SHIFT)
    # A recording that stalls is commoner than one whose crystal is out.
    assert not tab.fit_drift_check.isChecked()


def test_settings_are_forwarded_to_the_analysis(qtbot, tab, analysis, monkeypatch):
    used = {}

    def calculate(paths, offsets, progress, **settings):
        used.update(settings)
        return analysis

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.timing_correction.analyse_timing_corrections",
        calculate,
    )
    tab.window_spin.setValue(20.0)
    tab.search_spin.setValue(30.0)
    tab.min_quality_spin.setValue(4.5)
    tab.min_gap_spin.setValue(40.0)
    tab.fit_drift_check.setChecked(True)

    tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    # ``step`` is deliberately absent: drift_curve spaces windows at half a window.
    assert used == {
        "window": 20.0,
        "search": 30.0,
        "min_quality": 4.5,
        "min_shift": 0.04,  # the form is in milliseconds
        "fit_drift": True,
    }


def test_settings_are_locked_while_an_analysis_runs(qtbot, tab, analysis, monkeypatch):
    running = threading.Event()
    release = threading.Event()

    def calculate(paths, offsets, progress, **settings):
        running.set()
        release.wait(timeout=5.0)
        return analysis

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.timing_correction.analyse_timing_corrections",
        calculate,
    )

    tab.correct_button.click()
    qtbot.waitUntil(running.is_set)
    assert not tab.settings_widget.isEnabled()

    release.set()
    qtbot.waitUntil(lambda: not tab.is_busy())
    assert tab.settings_widget.isEnabled()
