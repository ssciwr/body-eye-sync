"""The tab base class, and the placeholder tabs that are nothing more than it."""

import pytest
from qtpy.QtWidgets import QLabel

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
    assert tab.done_button.text() == "Finish videos, align audio"
    assert tab.done_button.isDefault()


def test_alignment_tab_shows_paired_audio_after_video_step(qtbot, data_dir, tmp_path):
    path = data_dir / "three-people.mp4"
    audio = tmp_path / "mic1.wav"
    audio.touch()
    experiment = Experiment(ExperimentConfig())
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    mic = experiment.add_audio(AudioInput(id="mic1", path=audio, glasses_video="cam1"))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.done_button.click()
    assert tab.step_label.text().startswith("Step 2:")
    assert tab.audio_players[0].audio is mic
    assert tab.audio_players[0].button.text() == "Play mic1"


def test_alignment_tab_says_when_a_video_has_no_paired_audio(
    qtbot, experiment, data_dir
):
    path = data_dir / "three-people.mp4"
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.done_button.click()
    assert len(tab.no_paired_audio_labels) == 1
    assert tab.no_paired_audio_labels[0].text() == "No paired audio"
    assert not tab.no_paired_audio_labels[0].isEnabled()


def test_alignment_tab_shows_unpaired_audio_after_video_step(
    qtbot, experiment, tmp_path
):
    audio = tmp_path / "quartet.wav"
    audio.touch()
    experiment.add_audio(AudioInput(id="quartet-audio", path=audio))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.done_button.click()
    assert len(tab.audio_players) == 1
    assert tab.unpaired_audio_labels[0].text() == "Unpaired audio"
    assert tab.audio_players[0].button.text() == "Play quartet-audio"
