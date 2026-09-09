"""The tab base class, and the placeholder tabs that are nothing more than it."""

from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QLabel, QMessageBox, QPushButton

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
    TimelineConfig,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.media import VideoInfo
from body_eye_sync.gui.tabs import TAB_TYPES
from body_eye_sync.gui.tabs.alignment import AlignmentTab
from body_eye_sync.gui.tabs.base import BaseTab, PlaceholderTab
from body_eye_sync.gui.tabs.post_processing import PostProcessingTab

PLACEHOLDER_TABS = [
    PostProcessingTab,
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


def test_alignment_refresh_reuses_viewer_and_updates_controls(
    qtbot, experiment, data_dir, monkeypatch
):
    video = experiment.add_fixed_video(
        FixedVideoInput(id="room", path=data_dir / "three-people.mp4")
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    card = tab.video_cards[0]
    capture = card.viewer._capture
    card.viewer.set_frame(3)
    video.id = "renamed"
    video.timeline.offset = -0.04
    changes = []
    tab.experiment_changed.connect(lambda: changes.append(True))

    def unexpected(*args):
        pytest.fail("Unchanged video should not be stopped or reloaded")

    monkeypatch.setattr(card.viewer, "load", unexpected)
    monkeypatch.setattr(card.viewer, "stop", unexpected)
    tab.refresh()
    tab.refresh()

    assert tab.video_cards == [card]
    assert card.viewer._capture is capture
    assert card.input_label.text() == "renamed"
    assert card.controls.spin.value() == -0.04
    assert card.viewer.current_time_seconds == pytest.approx(0.04)
    assert card.shared_timeline_label.text() == "Shared Timeline point 0.000 s"
    assert not changes


def test_alignment_refresh_replaces_changed_inputs(
    qtbot, experiment, data_dir, distinct_videos
):
    room, other = distinct_videos(2)
    video = experiment.add_fixed_video(FixedVideoInput(id="room", path=room))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    original = tab.video_cards[0]
    extra = experiment.add_fixed_video(FixedVideoInput(id="extra", path=other))
    tab.refresh()
    assert tab.video_cards[0] is not original
    assert original.viewer._capture is None
    original = tab.video_cards[0]
    added = tab.video_cards[1]

    video.video_path = data_dir / "three-people-talking.mp4"
    tab.refresh()
    assert len(tab.video_cards) == 2
    assert tab.video_cards[0] is not original
    assert tab.video_cards[0].loaded_path == video.video_path
    assert original.viewer._capture is None
    assert added.viewer._capture is None
    original = tab.video_cards[0]

    experiment.remove_input(extra)
    tab.refresh()
    assert len(tab.video_cards) == 1
    assert tab.video_cards[0] is not original
    assert original.viewer._capture is None


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
    assert [card.input_label.text() for card in tab.video_cards] == [
        "cam1",
        "room0",
        "room1",
        "room2",
    ]
    button_row = tab.layout().itemAt(2).layout()
    assert not hasattr(tab, "estimate_button")
    assert tab.layout().itemAt(0).widget() is tab.align_button
    assert tab.align_button.text() == "Automatic alignment"
    assert button_row.indexOf(tab.reset_timeline_button) < button_row.indexOf(
        tab.play_all_button
    )
    assert not hasattr(tab, "shared_timeline_label")
    assert button_row.indexOf(tab.done_button) >= 0
    assert tab.done_button.text() == "Finish alignment"
    assert tab.layout().itemAt(1).widget() is tab.scroll_area
    assert tab.scroll_area.widget() is tab.video_grid_widget
    assert tab.done_button.isDefault()


def test_automatic_alignment_populates_offsets_for_manual_fine_tuning(
    qtbot, data_dir, monkeypatch
):
    path = data_dir / "three-people.mp4"
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room1", path=path),
                FixedVideoInput(id="room2", path=path),
            ]
        )
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    changed = []
    busy = []
    progress = []
    tab.experiment_changed.connect(lambda: changed.append(True))
    tab.busy_changed.connect(busy.append)
    tab.progress_changed.connect(lambda *values: progress.append(values))

    def align(current_experiment, *, progress):
        assert not tab.isEnabled()
        assert not tab.video_cards[0].viewer._play_button.isEnabled()
        assert not tab.video_cards[0].controls.spin.isEnabled()
        assert not tab.done_button.isEnabled()
        assert not tab.video_cards[0].viewer._timer.isActive()
        current_experiment.fixed_videos[0].timeline.offset = 0.125
        current_experiment.fixed_videos[1].timeline.offset = 0.375
        progress(0.5)
        return SimpleNamespace(offsets={"room1": 0.125, "room2": 0.375})

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.alignment.align_experiment",
        align,
    )

    tab.video_cards[0].viewer._play_button.setChecked(True)
    tab.align_button.click()

    assert [card.controls.spin.value() for card in tab.video_cards] == pytest.approx(
        [0.125, 0.375]
    )
    assert [
        card.viewer.current_time_seconds for card in tab.video_cards
    ] == pytest.approx([0.25, 0.0])
    assert [card.shared_timeline_label.text() for card in tab.video_cards] == [
        "Shared Timeline point 0.375 s",
        "Shared Timeline point 0.375 s",
    ]
    assert changed == [True]
    assert busy == [True, False]
    assert progress == [
        (0, 100, "Aligning recordings…"),
        (50, 100, "Aligning recordings…"),
    ]
    assert tab.align_button.isEnabled()
    assert tab.video_cards[0].viewer._play_button.isEnabled()
    assert tab.done_button.isEnabled()

    tab.video_cards[1].controls.up_button.click()

    assert experiment.fixed_videos[1].timeline.offset == pytest.approx(0.425)


def test_automatic_alignment_restores_controls_after_failure(
    qtbot, experiment, distinct_videos, monkeypatch
):
    for index, path in enumerate(distinct_videos(2)):
        experiment.add_fixed_video(FixedVideoInput(id=f"room{index}", path=path))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)

    def fail(*args, **kwargs):
        assert not tab.isEnabled()
        raise RuntimeError("Alignment failed")

    monkeypatch.setattr("body_eye_sync.gui.tabs.alignment.align_experiment", fail)
    with pytest.raises(RuntimeError, match="Alignment failed"):
        tab._align()

    assert tab.isEnabled()
    assert tab.align_button.isEnabled()
    assert tab.video_cards[0].viewer._play_button.isEnabled()


# Covers the video offset controls used during manual alignment.
def test_alignment_tab_edits_video_time_offset(
    qtbot, experiment, data_dir, distinct_videos, monkeypatch
):
    path = data_dir / "three-people.mp4"
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    experiment.add_fixed_video(FixedVideoInput(id="room1", path=distinct_videos(1)[0]))
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
    assert not hasattr(controls, "time_label")
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()
    assert layout.itemAt(0).widget() is controls.down_button
    assert layout.indexOf(controls.up_button) < layout.indexOf(controls.set_button)

    tab.video_cards[1].viewer.set_frame(3)

    controls.up_button.click()
    assert experiment.glasses_videos[0].timeline.offset == pytest.approx(0.05)
    assert tab.video_cards[0].viewer.current_frame < 0
    assert tab.video_cards[0].viewer.current_time_seconds == pytest.approx(-0.05)
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()

    controls.down_button.click()
    assert experiment.glasses_videos[0].timeline.offset == pytest.approx(0.0)
    assert tab.video_cards[0].viewer.current_frame == 0
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()

    controls.down_button.click()
    assert experiment.glasses_videos[0].timeline.offset == pytest.approx(-0.05)
    assert tab.video_cards[0].viewer.current_frame > 0

    tab.video_cards[0].viewer.set_frame(2)
    expected_offset = -2 / tab.video_cards[0].viewer._fps
    assert controls.set_button.property("needsOffset") is True
    assert controls.set_button.isEnabled()
    assert "#2563eb" in controls.set_button.styleSheet()
    controls.set_button.click()
    assert experiment.glasses_videos[0].timeline.offset == pytest.approx(
        expected_offset,
        abs=0.001,
    )
    assert controls.set_button.property("needsOffset") is False
    assert not controls.set_button.isEnabled()
    assert controls.spin.singleStep() == pytest.approx(0.05)
    assert experiment.fixed_videos[0].timeline.offset == pytest.approx(0.0)
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
            fixed_videos=[
                FixedVideoInput(
                    id="room1", path=path, timeline=TimelineConfig(offset=0.25)
                )
            ],
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
    assert experiment.glasses_videos[0].timeline.offset == pytest.approx(
        -source_time, abs=0.001
    )
    assert experiment.fixed_videos[0].timeline.offset == pytest.approx(-source_time)
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
    assert experiment.fixed_videos[0].timeline.offset == pytest.approx(0.0)
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
        FixedVideoInput(id="room1", path=path, timeline=TimelineConfig(offset=-0.12))
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.video_cards[0].viewer.set_frame(1)
    card = tab.video_cards[0]
    assert (
        card.shared_timeline_label.text() == "Before shared start time  (-0.080 s)"
    )  # shared time is -0.12 s offset + one 0.04 s frame.
    assert "#dc2626" in card.shared_timeline_label.styleSheet()
    tab.video_cards[0].viewer.set_frame(
        3
    )  # now we have gone 3 frames forward or 0.04*3 = 0.12
    assert card.shared_timeline_label.text() == "Shared Timeline point 0.000 s"
    tab.video_cards[0].viewer.set_frame(
        5
    )  # frame 5 clamps to frame 4 in this 5-frame video
    assert (
        card.shared_timeline_label.text() == "Shared Timeline point 0.040 s"
    )  # one frame ahead after the shared start.


"""
Note that here we do not ensure or test against that (A) the user cannot run "Set offset for all videos" if the
offset is longer than any of the videos (Those will be clamped to their last frame).
We also don't test that the final length for each video is > 0.
"""


def test_alignment_tab_play_all_uses_shared_timeline(
    qtbot, experiment, data_dir, distinct_videos, monkeypatch
):
    path = data_dir / "three-people.mp4"
    experiment.add_glasses_video(
        GlassesVideoInput(id="cam1", path=path, gaze_path=path.with_suffix(".tsv"))
    )
    experiment.add_fixed_video(
        FixedVideoInput(
            id="room1",
            path=distinct_videos(1)[0],
            timeline=TimelineConfig(offset=0.04),
        )
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    tab.video_cards[0].viewer.set_frame(1)
    secondary_audio_seeks = []
    monkeypatch.setattr(
        tab.video_cards[1].viewer,
        "_sync_audio_to_frame",
        lambda: secondary_audio_seeks.append(True),
    )
    tab.play_all_button.click()
    assert tab.video_cards[1].viewer.current_time_seconds == pytest.approx(0.0)
    monkeypatch.setattr(
        tab.video_cards[0].viewer, "_media_position_seconds", lambda: 0.08
    )
    tab.video_cards[0].viewer._advance()
    tab.play_all_button.click()
    assert tab.video_cards[0].viewer.current_time_seconds == pytest.approx(0.08)
    assert tab.video_cards[1].viewer.current_time_seconds == pytest.approx(0.04)
    assert secondary_audio_seeks == []


def test_alignment_tab_uses_corrected_clock_rates(qtbot, experiment, distinct_videos):
    slow, steady = distinct_videos(2)
    experiment.add_fixed_video(
        FixedVideoInput(
            id="slow",
            path=slow,
            timeline=TimelineConfig(offset=0.0, rate=2.0),
        )
    )
    experiment.add_fixed_video(FixedVideoInput(id="steady", path=steady))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    slow, steady = tab.video_cards

    tab._show_shared_timeline_time(0.08)

    assert slow.viewer.current_time_seconds == pytest.approx(0.04)
    assert steady.viewer.current_time_seconds == pytest.approx(0.08)

    slow.controls.up_button.click()
    assert slow.video.timeline.offset == pytest.approx(0.05)
    assert slow.viewer.current_time_seconds == pytest.approx(0.015)
    slow.controls.down_button.click()

    slow.viewer.set_time_seconds(0.04, show_requested_time=True)
    tab.play_all_button.click()

    assert steady.viewer.current_time_seconds == pytest.approx(0.08)
    assert steady.shared_timeline_label.text() == "Shared Timeline point 0.080 s"
    tab.play_all_button.click()


def test_alignment_tab_play_all_preserves_exact_start_across_frame_rates(
    qtbot, experiment, distinct_videos
):
    first, second = distinct_videos(2)
    experiment.add_fixed_video(FixedVideoInput(id="room1", path=first))
    experiment.add_fixed_video(FixedVideoInput(id="room2", path=second))
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    primary = tab.video_cards[0].viewer
    secondary = tab.video_cards[1].viewer
    # A frame rate belongs to the file, not to the viewer showing it, so the
    # second rate is stubbed on the video: 10 fps starting where the file does.
    secondary._video._info = VideoInfo(frame_rate=10.0)
    secondary._video._info_path = secondary._video.video_path
    primary.set_time_seconds(0.05, show_requested_time=True)

    tab.play_all_button.click()

    assert primary.current_time_seconds == pytest.approx(0.05)
    assert primary.current_media_time_seconds == pytest.approx(0.04)
    assert secondary.current_time_seconds == pytest.approx(0.05)
    assert secondary.current_media_time_seconds == pytest.approx(0.0)
    assert tab.video_cards[1].shared_timeline_label.text() == (
        "Shared Timeline point 0.050 s"
    )
    tab.play_all_button.click()


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


def test_alignment_tab_finish_stops_all_video_playback(qtbot, experiment, data_dir):
    experiment.add_fixed_video(
        FixedVideoInput(id="room1", path=data_dir / "three-people.mp4")
    )
    tab = AlignmentTab(experiment)
    qtbot.addWidget(tab)
    viewer = tab.video_cards[0].viewer
    viewer._play_button.click()
    assert viewer._timer.isActive()

    tab.done_button.click()

    assert not viewer._timer.isActive()
    assert not viewer._play_button.isChecked()
