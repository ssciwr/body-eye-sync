import threading

import pytest

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
    TimelineConfig,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.gui.tabs.clock_rate import ClockRateTab
from body_eye_sync.preprocessing.clock_rate import (
    MIN_DRIFT_PPM,
    DEFAULT_SEARCH,
    DEFAULT_WINDOW,
    SPECTRAL_MIN_QUALITY,
    OffsetPoint,
    ClockRateAnalysis,
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
                    timeline=TimelineConfig(offset=5.0),
                )
            ],
        )
    )


@pytest.fixture
def tab(qtbot, experiment):
    widget = ClockRateTab(experiment)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def analysis():
    return ClockRateAnalysis(
        reference="room",
        points={
            "room": [],
            "glasses": [
                OffsetPoint(60.0, 5.2),
                OffsetPoint(120.0, 5.3),
            ],
        },
        fits={
            "glasses": Timeline(offset=5.2, rate=1.0000182),
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

    assert headers == ["Id", "Offset", "Clock drift"]
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
    assert [tab.table.item(glasses, column).text() for column in range(3)] == [
        "glasses",
        "+5.000 s",
        "0",
    ]
    assert tab.correct_button.text() == "Analyse and correct clock rates"
    assert tab.clear_button.text() == "Clear corrections"


def test_existing_corrections_are_plotted_as_lines_without_data_points(tab):
    glasses = tab.experiment.glasses_videos[0]
    glasses.timeline.rate = 1.0000182

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
        "body_eye_sync.gui.tabs.clock_rate.analyse_clock_rates",
        calculate,
    )

    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    glasses = tab.experiment.glasses_videos[0]
    assert measured_offsets == [{"glasses": 5.0, "room": 0.0}]
    assert glasses.timeline.offset == pytest.approx(5.2)
    assert glasses.timeline.rate == pytest.approx(1.0000182)
    row = _row(tab, "glasses")
    assert [tab.table.item(row, column).text() for column in range(1, 3)] == [
        "+5.200 s",
        "+18.2 ppm",
    ]
    assert tab.canvas.isVisibleTo(tab)
    assert tab.figure.axes[0].lines
    assert tab.figure.axes[0].collections


def test_only_a_significant_drift_fit_is_applied(qtbot, tab, monkeypatch):
    analysis = ClockRateAnalysis(
        reference="glasses",
        points={"glasses": [], "room": [OffsetPoint(60.0, 0.4)]},
        fits={"glasses": Timeline(offset=5.2, rate=1.0000182)},
        unavailable=[],
    )
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.clock_rate.analyse_clock_rates",
        lambda paths, offsets, progress, **settings: analysis,
    )
    messages = []
    tab.status_message.connect(messages.append)

    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    room = tab.experiment.fixed_videos[0]
    assert room.timeline.offset == 0.0
    assert room.timeline.rate == 1.0
    assert tab.experiment.glasses_videos[0].timeline.offset == pytest.approx(5.2)
    assert messages == ["Updated the clock rate of 1 input(s)"]


def test_no_significant_drift_keeps_the_alignment_offset_and_still_plots(
    qtbot, tab, monkeypatch
):
    analysis = ClockRateAnalysis(
        reference="room",
        points={"room": [], "glasses": [OffsetPoint(60.0, 4.75)]},
        fits={},
        unavailable=[],
    )
    changed = []
    tab.experiment_changed.connect(lambda: changed.append(True))
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.clock_rate.analyse_clock_rates",
        lambda paths, offsets, progress, **settings: analysis,
    )
    messages = []
    tab.status_message.connect(messages.append)

    tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    assert messages == ["No clock-rate changes detected"]
    # The measurements are worth seeing whether or not they moved anything.
    assert tab.canvas.isVisibleTo(tab)
    assert tab.figure.axes[0].collections
    assert (
        tab.figure.axes[0].get_title()
        == "Measured offsets: no clock-rate correction needed"
    )
    assert tab.experiment.glasses_videos[0].timeline.offset == pytest.approx(5.0)
    assert tab.experiment.glasses_videos[0].timeline.rate == 1.0
    assert changed == []


def test_correction_requires_two_inputs(qtbot, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
            ]
        )
    )
    tab = ClockRateTab(experiment)
    qtbot.addWidget(tab)

    assert not tab.correct_button.isEnabled()


def test_clear_is_disabled_until_there_is_something_to_clear(tab):
    assert not tab.clear_button.isEnabled()

    tab.experiment.glasses_videos[0].timeline.rate = 1.0000182
    tab.refresh()

    assert tab.clear_button.isEnabled()


def test_clear_resets_clock_rates_but_keeps_the_offsets(qtbot, tab):
    glasses = tab.experiment.glasses_videos[0]
    glasses.timeline.rate = 1.0000182
    tab.refresh()
    messages = []
    tab.status_message.connect(messages.append)

    with qtbot.waitSignal(tab.experiment_changed):
        tab.clear_button.click()

    assert glasses.timeline.rate == 1.0
    # Where the recording starts is alignment's answer, not this tab's to undo.
    assert glasses.timeline.offset == 5.0
    row = _row(tab, "glasses")
    assert [tab.table.item(row, column).text() for column in range(1, 3)] == [
        "+5.000 s",
        "0",
    ]
    assert messages == ["Clock-rate corrections cleared"]
    assert not tab.canvas.isVisibleTo(tab)
    assert not tab.clear_button.isEnabled()


def test_clear_drops_a_fresh_analysis_plot(qtbot, tab, analysis, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.clock_rate.analyse_clock_rates",
        lambda paths, offsets, progress, **settings: analysis,
    )
    with qtbot.waitSignal(tab.experiment_changed):
        tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())
    assert tab.canvas.isVisibleTo(tab)

    with qtbot.waitSignal(tab.experiment_changed):
        tab.clear_button.click()

    assert tab.experiment.glasses_videos[0].timeline.rate == 1.0
    # The corrected offset stays: clearing does not re-run alignment.
    assert tab.experiment.glasses_videos[0].timeline.offset == pytest.approx(5.2)
    assert not tab.canvas.isVisibleTo(tab)


def test_settings_default_to_the_analysis_defaults(tab):
    assert tab.window_spin.value() == pytest.approx(DEFAULT_WINDOW)
    assert tab.search_spin.value() == pytest.approx(DEFAULT_SEARCH)
    assert tab.min_quality_spin.value() == pytest.approx(SPECTRAL_MIN_QUALITY)
    assert tab.min_drift_spin.value() == pytest.approx(MIN_DRIFT_PPM)


def test_settings_are_forwarded_to_the_analysis(qtbot, tab, analysis, monkeypatch):
    used = {}

    def calculate(paths, offsets, progress, **settings):
        used.update(settings)
        return analysis

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.clock_rate.analyse_clock_rates",
        calculate,
    )
    tab.window_spin.setValue(20.0)
    tab.search_spin.setValue(30.0)
    tab.min_quality_spin.setValue(4.5)
    tab.min_drift_spin.setValue(4.0)

    tab.correct_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    assert used == {
        "window": 20.0,
        "search": 30.0,
        "min_quality": 4.5,
        "min_drift_ppm": 4.0,
    }


def test_settings_are_locked_while_an_analysis_runs(qtbot, tab, analysis, monkeypatch):
    running = threading.Event()
    release = threading.Event()

    def calculate(paths, offsets, progress, **settings):
        running.set()
        release.wait(timeout=5.0)
        return analysis

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.clock_rate.analyse_clock_rates",
        calculate,
    )

    tab.correct_button.click()
    qtbot.waitUntil(running.is_set)
    assert not tab.settings_widget.isEnabled()

    release.set()
    qtbot.waitUntil(lambda: not tab.is_busy())
    assert tab.settings_widget.isEnabled()
