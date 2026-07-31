"""The tab base class, and the placeholder tabs that are nothing more than it."""

import pytest
from qtpy.QtWidgets import QLabel, QMessageBox, QPushButton

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs import TAB_TYPES
from body_eye_sync.gui.tabs.alignment import AlignmentTab
from body_eye_sync.gui.tabs.audio_processing import AudioProcessingTab
from body_eye_sync.gui.tabs.base import BaseTab, PlaceholderTab
from body_eye_sync.gui.tabs.data_export import DataExportTab
from body_eye_sync.gui.tabs.post_processing import PostProcessingTab

PLACEHOLDER_TABS = [
    AudioProcessingTab,
    PostProcessingTab,
    DataExportTab,
]


@pytest.fixture
def experiment():
    return Experiment(ExperimentConfig())


def test_every_tab_has_a_title():
    assert all(tab_type.title for tab_type in TAB_TYPES)


def test_a_tab_holds_the_experiment_it_is_given(qtbot, experiment):
    tab = BaseTab(experiment)
    qtbot.addWidget(tab)

    assert tab.experiment is experiment


def test_set_experiment_swaps_it_and_refreshes(qtbot, experiment):
    class _Tab(BaseTab):
        refreshed = 0

        def refresh(self):
            self.refreshed += 1

    tab = _Tab(experiment)
    qtbot.addWidget(tab)
    other = Experiment(ExperimentConfig())

    tab.set_experiment(other)

    assert tab.experiment is other
    assert tab.refreshed == 1


@pytest.mark.parametrize("tab_type", PLACEHOLDER_TABS, ids=lambda t: t.__name__)
def test_placeholder_tabs_say_so(qtbot, experiment, tab_type):
    tab = tab_type(experiment)
    qtbot.addWidget(tab)

    assert issubclass(tab_type, PlaceholderTab)
    label = tab.findChild(QLabel)
    assert label.text() == f"{tab_type.title} is not implemented yet"


def test_alignment_tab_renders_all_videos_without_overlays(qtbot, experiment, data_dir):
    path = data_dir / "three-people.mp4"
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1", path=path, gaze_path=path.with_suffix(".tsv")
                )
            ],
            fixed_videos=[FixedVideoInput(id=f"room{i}", path=path) for i in range(3)],
        )
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    assert len(tab.video_viewers) == 4
    assert tab.grid.itemAtPosition(1, 0).widget() is tab._cells[3]
    assert all(not viewer.show_overlays for viewer in tab.video_viewers)
    button_row = tab.layout().itemAt(1).layout()
    assert tab.estimate_button.text() == "Estimate inter-video offsets automatically"
    assert button_row.indexOf(tab.estimate_button) < button_row.indexOf(tab.done_button)
    assert tab.done_button.text() == "Finish alignment"
    assert tab.layout().itemAt(0).layout() is tab.grid
    assert tab.done_button.isDefault()


# Covers the video offset controls used during manual alignment.
def test_alignment_tab_edits_video_time_offset(
    qtbot, experiment, data_dir, monkeypatch
):
    path = data_dir / "three-people.mp4"
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    experiment.add_fixed_video(FixedVideoInput(id="room1", path=path))
    changed = []
    tab = AlignmentTab(experiment)
    tab.experiment_changed.connect(lambda: changed.append(True))
    qtbot.addWidget(tab)
    controls = tab.video_controls[0]
    questions = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: questions.append(args) or QMessageBox.StandardButton.No,
    )

    layout = controls.layout()
    assert controls.set_button.text() == "Set"
    assert controls.set_button.toolTip() == "Set as offset"
    assert controls.set_button.property("needsOffset") is False
    assert layout.indexOf(controls.up_button) < layout.indexOf(controls.set_button)

    tab.video_viewers[1].set_frame(3)

    controls.up_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(0.05)
    assert tab.video_viewers[0].current_frame < 0
    assert tab.video_viewers[0].current_time_seconds == pytest.approx(-0.05)
    assert controls.set_button.property("needsOffset") is False

    controls.down_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(0.0)
    assert tab.video_viewers[0].current_frame == 0
    assert controls.set_button.property("needsOffset") is False

    controls.down_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(-0.05)
    assert tab.video_viewers[0].current_frame > 0

    tab.video_viewers[0].set_frame(2)
    assert controls.set_button.property("needsOffset") is True
    assert "#2563eb" in controls.set_button.styleSheet()
    controls.set_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(
        -2 / tab.video_viewers[0]._fps,
        abs=0.001,
    )
    assert controls.set_button.property("needsOffset") is False
    assert controls.set_button.styleSheet() == ""
    assert controls.spin.singleStep() == pytest.approx(0.05)
    assert experiment.fixed_videos[0].time_offset == pytest.approx(0.0)
    assert questions[0][1] == "Set other videos?"
    assert len(changed) == 4


def test_alignment_tab_can_set_other_videos_to_same_video_timestamp(
    qtbot, data_dir, monkeypatch
):
    path = data_dir / "three-people.mp4"
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1", path=path, gaze_path=path.with_suffix(".tsv")
                )
            ],
            fixed_videos=[FixedVideoInput(id="room1", path=path)],
        )
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)

    tab.video_viewers[0].set_frame(2)
    tab.video_viewers[1].set_frame(3)
    source_time = tab.video_viewers[0].current_time_seconds
    tab.video_controls[0].set_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(
        -2 / tab.video_viewers[0]._fps, abs=0.001
    )
    assert experiment.fixed_videos[0].time_offset == pytest.approx(
        experiment.glasses_videos[0].time_offset
    )
    assert tab.video_viewers[1].current_time_seconds == pytest.approx(source_time)


# Covers applying every automatic glasses-video offset proposal.
def test_alignment_tab_estimates_glasses_offsets(qtbot, data_dir, monkeypatch):
    path = data_dir / "three-people.mp4"
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1", path=path, gaze_path=path.with_suffix(".tsv")
                ),
                GlassesVideoInput(
                    id="cam2", path=path, gaze_path=path.with_suffix(".tsv")
                ),
            ],
            fixed_videos=[FixedVideoInput(id="room1", path=path, time_offset=9.0)],
        )
    )
    experiment.glasses_videos[0].time_offset = 4.0
    experiment.glasses_videos[1].time_offset = 5.0
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.alignment.assign_automatic_estimated_offset",
        lambda *videos: {videos[0]: 0.0, videos[1]: -1.25},
    )
    changed = []
    statuses = []
    tab = AlignmentTab(experiment)
    tab.experiment_changed.connect(lambda: changed.append(True))
    tab.status_message.connect(statuses.append)
    qtbot.addWidget(tab)

    tab.estimate_button.click()

    assert experiment.glasses_videos[0].time_offset == 0.0
    assert experiment.glasses_videos[1].time_offset == pytest.approx(-1.25)
    assert experiment.fixed_videos[0].time_offset == pytest.approx(9.0)
    assert statuses[-1] == "Estimated offsets: 2 changed"
    assert changed


# Covers finishing video alignment without opening audio controls.
def test_alignment_tab_finish_emits_finished_without_audio_controls(
    qtbot, data_dir, tmp_path
):
    path = data_dir / "three-people.mp4"
    audio = tmp_path / "mic1.wav"
    audio.touch()
    experiment = Experiment(ExperimentConfig())
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    experiment.add_audio(AudioInput(id="mic1", path=audio, glasses_video="cam1"))
    tab = AlignmentTab(experiment)
    finished = []
    tab.finished.connect(lambda: finished.append(True))
    qtbot.addWidget(tab)

    tab.done_button.click()

    assert finished == [True]
    assert tab.done_button.text() == "Finish alignment"
    assert "Play mic1" not in [
        button.text() for button in tab.findChildren(QPushButton)
    ]
