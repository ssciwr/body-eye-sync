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
    assert len(tab.video_cards) == 4
    assert tab.grid.itemAtPosition(1, 0).widget() is tab.video_cards[3]
    assert all(not card.viewer.show_overlays for card in tab.video_cards)
    button_row = tab.layout().itemAt(1).layout()
    assert not hasattr(tab, "estimate_button")
    assert button_row.indexOf(tab.reset_timeline_button) < button_row.indexOf(
        tab.play_all_button
    )
    assert not hasattr(tab, "shared_timeline_label")
    assert button_row.indexOf(tab.done_button) >= 0
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
    controls = tab.video_cards[0].controls
    dialogs = []
    clicked = {}

    def click_this_video(dialog):
        dialogs.append(dialog)
        clicked[id(dialog)] = next(
            button for button in dialog.buttons() if button.text() == "This video"
        )

    monkeypatch.setattr(QMessageBox, "exec", click_this_video)
    monkeypatch.setattr(
        QMessageBox, "clickedButton", lambda dialog: clicked[id(dialog)]
    )

    layout = controls.layout()
    assert controls.set_button.text() == "Zero here"
    assert controls.set_button.toolTip() == (
        "Set offset so this frame is timeline zero"
    )
    assert not hasattr(controls, "time_label")
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()
    assert layout.itemAt(0).widget() is controls.down_button
    assert layout.indexOf(controls.up_button) < layout.indexOf(controls.set_button)

    tab.video_cards[1].viewer.set_frame(3)

    controls.up_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(0.05)
    assert tab.video_cards[0].viewer.current_frame < 0
    assert tab.video_cards[0].viewer.current_time_seconds == pytest.approx(-0.05)
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()

    controls.down_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(0.0)
    assert tab.video_cards[0].viewer.current_frame == 0
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()

    controls.down_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(-0.05)
    assert tab.video_cards[0].viewer.current_frame > 0

    tab.video_cards[0].viewer.set_frame(2)
    expected_offset = -2 / tab.video_cards[0].viewer._fps
    assert controls.set_button.property("needsOffset") is True
    assert controls.set_button.isEnabled()
    assert "#2563eb" in controls.set_button.styleSheet()
    controls.set_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(
        expected_offset,
        abs=0.001,
    )
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()
    assert controls.spin.singleStep() == pytest.approx(0.05)
    assert experiment.fixed_videos[0].time_offset == pytest.approx(0.0)
    assert tab.video_cards[1].viewer.current_time_seconds == pytest.approx(0.0)
    assert tab.video_cards[0].shared_timeline_label.text() == (
        "Shared Timeline point 0.000 s"
    )
    assert tab.video_cards[1].shared_timeline_label.text() == (
        "Shared Timeline point 0.000 s"
    )
    assert dialogs[0].text() == (
        f"Apply offset {expected_offset:.3f} s to this video, or to all videos?"
    )
    assert dialogs[0].defaultButton().text() == "This video"
    assert len(changed) == 4


def test_alignment_tab_applies_zero_offset_to_all_videos(qtbot, data_dir, monkeypatch):
    # Check that
    path = data_dir / "three-people.mp4"
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1", path=path, gaze_path=path.with_suffix(".tsv")
                )
            ],
            fixed_videos=[FixedVideoInput(id="room1", path=path, time_offset=0.25)],
        )
    )
    clicked = {}

    def click_all_videos(dialog):
        clicked[id(dialog)] = next(
            button for button in dialog.buttons() if button.text() == "All videos"
        )

    monkeypatch.setattr(QMessageBox, "exec", click_all_videos)
    monkeypatch.setattr(
        QMessageBox, "clickedButton", lambda dialog: clicked[id(dialog)]
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)

    tab.video_cards[0].viewer.set_frame(2)
    tab.video_cards[1].viewer.set_frame(3)
    source_time = tab.video_cards[0].viewer.current_time_seconds
    tab.video_cards[0].controls.set_button.click()
    assert experiment.glasses_videos[0].time_offset == pytest.approx(
        -source_time, abs=0.001
    )
    assert experiment.fixed_videos[0].time_offset == pytest.approx(-source_time)
    assert tab.video_cards[1].viewer.current_time_seconds == pytest.approx(source_time)
    assert tab.video_cards[0].shared_timeline_label.text() == (
        "Shared Timeline point 0.000 s"
    )
    assert tab.video_cards[1].shared_timeline_label.text() == (
        "Shared Timeline point 0.000 s"
    )


def test_alignment_zero_button_ignores_closed_dialog(
    qtbot, experiment, data_dir, monkeypatch
):
    # closing the dialog does not apply the offset.
    experiment.add_fixed_video(
        FixedVideoInput(id="room1", path=data_dir / "three-people.mp4")
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    monkeypatch.setattr(QMessageBox, "exec", lambda _dialog: None)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda _dialog: None)
    card = tab.video_cards[0]
    card.viewer.set_frame(2)
    card.controls.set_button.click()
    assert experiment.fixed_videos[0].time_offset == pytest.approx(0.0)
    assert card.controls.set_button.property("needsOffset") is True


"""
Do we calculate the correct shared timeline based on moving 1 frame forward (1/25th of a second --> 0.04)
AKA does the video_viewers set_frame(1) method of moving forward correctly align with the timeline / offset values?

This uses three_people.mp4 added by Liam (But any 25 fps video will work with the below test - I chose to hardcode
the fps of the video rather than extract/calculate it as that goes beyond the scope of this test.
"""


def test_alignment_tab_marks_negative_shared_timeline_preview(
    qtbot, experiment, data_dir
):

    path = data_dir / "three-people.mp4"
    experiment.add_fixed_video(
        FixedVideoInput(id="room1", path=path, time_offset=-0.12)
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.video_cards[0].viewer.set_frame(1)
    card = tab.video_cards[0]
    assert (
        card.shared_timeline_label.text() == "Shared Timeline point -0.080 s"
    )  # this is 0.12 - 0.04 (1 Frames duration)
    assert card._pre_shared_overlay.isVisible()
    assert not card._shared_start_marker.isHidden()
    tab.video_cards[0].viewer.set_frame(
        3
    )  # now we have gone 3 frames forward or 0.04*3 = 0.12
    assert card.shared_timeline_label.text() == "Shared Timeline point 0.000 s"
    tab.video_cards[0].viewer.set_frame(
        5
    )  # now we have gone 3 frames forward or 0.04*3 = 0.12
    assert (
        card.shared_timeline_label.text() == "Shared Timeline point 0.080 s"
    )  # the inversion of the above, 2 frames ahead.


"""
Note that here we do not ensure or test against that (A) the user cannot run "Set offset for all videos" if the
offset is longer than any of the videos (Those will be clamped to their last frame).
We also don't test that the final length for each video is > 0.
"""


def test_alignment_tab_play_all_uses_shared_timeline(qtbot, experiment, data_dir):
    path = data_dir / "three-people.mp4"
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    experiment.add_fixed_video(FixedVideoInput(id="room1", path=path, time_offset=0.04))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.video_cards[0].viewer.set_frame(1)
    tab.play_all_button.click()
    assert tab.video_cards[1].viewer.current_time_seconds == pytest.approx(0.0)
    tab.video_cards[0].viewer._advance()
    tab.play_all_button.click()
    assert tab.video_cards[0].viewer.current_time_seconds == pytest.approx(0.08)
    assert tab.video_cards[1].viewer.current_time_seconds == pytest.approx(0.04)


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
