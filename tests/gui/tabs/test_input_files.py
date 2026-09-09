import json
from pathlib import Path

import pytest
from qtpy.QtWidgets import QWidget

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import ExperimentConfig
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo
from body_eye_sync.gui.tabs.input_files import (
    GAZE_FILE_ACTION,
    GAZE_FOLDER_ACTION,
    InputFilesTab,
)

_ID, _FILE, _EXTRA = range(3)
_GAZE = _GLASSES = _EXTRA


@pytest.fixture
def experiment():
    return Experiment(ExperimentConfig())


@pytest.fixture
def tab(qtbot, experiment):
    tab = InputFilesTab(experiment)
    qtbot.addWidget(tab)
    return tab


@pytest.fixture
def changes(tab):
    """Counts the ``experiment_changed`` signals the tab emits."""
    emitted = []
    tab.experiment_changed.connect(lambda: emitted.append(True))
    return emitted


@pytest.fixture
def messages(tab):
    """Collects the tab's status bar messages."""
    emitted = []
    tab.status_message.connect(emitted.append)
    return emitted


#: The header a device writes its gaze export with; see tests/data/three-people.tsv.
GAZE_HEADER = "participant\trectimestamp\tgaze_x\tgaze_y\tpupil_left\tpupil_right\tgaze_video_time\n"


def _write_gaze(path):
    """A stand-in gaze export: the real header and one sample."""
    path.write_text(f"{GAZE_HEADER}1403\t5197.003\t1385\t995\t4021\t4091\t0\n")
    return path


def _glasses_video(folder, name="cam1"):
    """A glasses video path with its gaze file beside it, as a device exports them."""
    _write_gaze(folder / f"{name}.tsv")
    return folder / f"{name}.mp4"


def _gaze_action(section, row, label):
    """The action of a row's gaze menu that picks a source of that kind."""
    button = section.table.cellWidget(row, _GAZE)
    return next(action for action in button.menu().actions() if action.text() == label)


def _answer_folder_dialog(monkeypatch, *chosen):
    """Answer the folder chooser with ``chosen`` (nothing = cancelled)."""
    asked = []

    def choose_folders(_parent, title):
        asked.append(title)
        return [Path(folder) for folder in chosen if str(folder)]

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.choose_folders", choose_folders
    )
    return asked


def _answer_gaze_folder_dialog(monkeypatch, chosen):
    """Answer the "recording folder for …" chooser, which takes one folder."""
    asked = []

    def get_existing_directory(_parent, title, *args, **kwargs):
        asked.append(title)
        return str(chosen)

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.QFileDialog.getExistingDirectory",
        get_existing_directory,
    )
    return asked


def _headers(section):
    header = section.table.horizontalHeader()
    return [section.table.horizontalHeaderItem(i).text() for i in range(header.count())]


def _cells(section, row):
    return [
        section.table.item(row, column).text()
        for column in range(section.table.columnCount())
        if section.table.item(row, column) is not None
    ]


def test_a_section_per_input_type_starting_with_glasses_videos(tab):
    assert [section.title() for section in tab.sections] == [
        "Glasses videos",
        "Fixed videos",
        "Audio",
    ]


def test_an_empty_section_says_so_instead_of_showing_a_table(tab):
    section = tab.glasses_section
    layout = section.layout()
    actions = layout.itemAt(2).layout()

    assert section.table.rowCount() == 0
    assert not section.table.isVisibleTo(section)
    assert section.empty_label.isVisibleTo(section)
    assert section.empty_label.text() == "No glasses videos yet"
    assert not section.remove_button.isEnabled()
    assert layout.itemAt(0).widget() is section.empty_label
    assert layout.itemAt(1).widget() is section.table
    # A whole recording is the better thing to add, so it is offered first.
    assert actions.itemAt(1).widget() is section.add_folder_button
    assert actions.itemAt(2).widget() is section.add_button


def test_add_glasses_video_uses_the_filename_as_its_id(tab, changes, data_dir):
    video = data_dir / "three-people.mp4"

    tab.glasses_section.add_files([video])

    assert [v.id for v in tab.experiment.glasses_videos] == ["three-people"]
    assert tab.experiment.glasses_videos[0].video_path == video
    assert _cells(tab.glasses_section, 0) == ["three-people", str(video)]
    assert tab.glasses_section.table.isVisibleTo(tab.glasses_section)
    assert not tab.glasses_section.empty_label.isVisibleTo(tab.glasses_section)
    assert changes == [True]


def test_each_input_goes_in_its_own_section(tab, data_dir, tmp_path):
    tab.audio_section.add_files([tmp_path / "mic1.wav"])
    tab.fixed_section.add_files([tmp_path / "room.mp4"])
    tab.glasses_section.add_files([data_dir / "three-people.mp4"])

    assert [type(i) for i in tab.experiment.inputs] == [GlassesVideo, FixedVideo, Audio]
    assert [section.table.rowCount() for section in tab.sections] == [1, 1, 1]
    assert _cells(tab.fixed_section, 0)[0] == "room"
    assert _cells(tab.audio_section, 0)[0] == "mic1"


def test_each_type_has_the_extra_columns_it_needs(tab, tmp_path):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    tab.fixed_section.add_files([tmp_path / "room.mp4"])
    tab.audio_section.add_files([tmp_path / "mic1.wav"])

    # An extra column, and its widget, is only where it means something.
    # The glasses section has two file columns, so each says which it is.
    assert _headers(tab.glasses_section) == ["Id", "Video file", "Gaze file"]
    assert _headers(tab.fixed_section) == ["Id", "File"]
    assert _headers(tab.audio_section) == ["Id", "File", "Glasses video"]
    assert tab.glasses_section.table.cellWidget(0, _GAZE) is not None
    assert tab.fixed_section.table.cellWidget(0, _EXTRA) is None
    assert tab.audio_section.table.cellWidget(0, _GLASSES) is not None


def test_a_section_is_only_as_tall_as_its_rows(tab, tmp_path):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    one_row = tab.glasses_section.table.height()

    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam2")])

    assert tab.glasses_section.table.height() > one_row


def test_adding_the_same_filename_twice_gives_unique_ids(tab, data_dir, tmp_path):
    other = tmp_path / "three-people.mp4"
    other.write_bytes(b"")

    tab.glasses_section.add_files([data_dir / "three-people.mp4"])
    tab.fixed_section.add_files([other])

    # Ids are unique across the types, not just within a section.
    assert [i.id for i in tab.experiment.inputs] == ["three-people", "three-people-2"]


def _answer_gaze_dialog(monkeypatch, chosen):
    """Answer the "gaze file for …" dialog with ``chosen`` (empty = cancelled)."""
    asked = []

    def get_open_file_name(_parent, title, *args, **kwargs):
        asked.append(title)
        return str(chosen), ""

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.QFileDialog.getOpenFileName",
        get_open_file_name,
    )
    return asked


def test_a_glasses_video_takes_the_gaze_file_beside_it(tab, data_dir, monkeypatch):
    # A device exports the gaze samples next to the video, so nothing is asked.
    asked = _answer_gaze_dialog(monkeypatch, "")

    tab.glasses_section.add_files([data_dir / "three-people.mp4"])

    assert tab.experiment.glasses_videos[0].gaze_path == data_dir / "three-people.tsv"
    assert asked == []
    assert tab.glasses_section.table.cellWidget(0, _GAZE).text() == "three-people.tsv"


def test_a_glasses_video_without_one_beside_it_asks(tab, tmp_path, monkeypatch):
    gaze = tmp_path / "elsewhere" / "gaze.tsv"
    gaze.parent.mkdir()
    _write_gaze(gaze)
    asked = _answer_gaze_dialog(monkeypatch, gaze)

    tab.glasses_section.add_files([tmp_path / "cam1.mp4"])

    assert asked == ["Gaze file for cam1.mp4"]
    assert tab.experiment.glasses_videos[0].gaze_path == gaze


def test_a_glasses_video_is_not_added_without_a_gaze_file(
    tab, changes, messages, tmp_path, monkeypatch
):
    _answer_gaze_dialog(monkeypatch, "")  # the user cancelled the dialog

    tab.glasses_section.add_files([tmp_path / "cam1.mp4"])

    assert tab.experiment.glasses_videos == []
    assert messages == ["cam1.mp4 not added: it needs a gaze file"]
    assert changes == []


def test_the_gaze_file_can_be_changed(tab, changes, tmp_path, monkeypatch):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    other = _write_gaze(tmp_path / "corrected.tsv")
    _answer_gaze_dialog(monkeypatch, other)
    changes.clear()

    _gaze_action(tab.glasses_section, 0, GAZE_FILE_ACTION).trigger()

    assert tab.experiment.glasses_videos[0].gaze_path == other
    assert tab.glasses_section.table.cellWidget(0, _GAZE).text() == "corrected.tsv"
    assert changes == [True]


def test_a_missing_gaze_file_is_flagged(tab, tmp_path, monkeypatch):
    _answer_gaze_dialog(monkeypatch, tmp_path / "never-exported.tsv")

    tab.glasses_section.add_files([tmp_path / "cam1.mp4"])

    button = tab.glasses_section.table.cellWidget(0, _GAZE)
    assert button.text() == "never-exported.tsv (not found)"


def test_a_missing_file_is_flagged(tab, tmp_path):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "gone")])

    assert tab.glasses_section.table.item(0, _FILE).text().endswith("(not found)")


def test_renaming_a_row_renames_the_input(tab, changes, data_dir):
    tab.glasses_section.add_files([data_dir / "three-people.mp4"])
    changes.clear()

    tab.glasses_section.table.item(0, _ID).setText("cam1")

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam1"]
    assert changes == [True]


@pytest.mark.parametrize("new_id", ["room", "", "sub/cam1"])
def test_an_invalid_id_is_reported_and_reverted(tab, messages, tmp_path, new_id):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    tab.fixed_section.add_files([tmp_path / "room.mp4"])

    tab.glasses_section.table.item(0, _ID).setText(new_id)

    assert [i.id for i in tab.experiment.inputs] == ["cam1", "room"]
    assert tab.glasses_section.table.item(0, _ID).text() == "cam1"
    assert len(messages) == 1
    assert "Could not rename input" in messages[0]


def test_audio_can_be_pointed_at_a_glasses_video(tab, tmp_path):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    tab.audio_section.add_files([tmp_path / "mic1.wav"])
    combo = tab.audio_section.table.cellWidget(0, _GLASSES)
    assert [combo.itemText(i) for i in range(combo.count())] == ["—", "cam1"]

    combo.setCurrentIndex(1)

    assert tab.experiment.audio[0].glasses_video is tab.experiment.glasses_videos[0]


def test_a_new_glasses_video_becomes_a_choice_for_the_audio(tab, tmp_path):
    tab.audio_section.add_files([tmp_path / "mic1.wav"])

    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])

    combo = tab.audio_section.table.cellWidget(0, _GLASSES)
    assert [combo.itemText(i) for i in range(combo.count())] == ["—", "cam1"]


def test_selecting_in_one_section_clears_the_others(tab, tmp_path):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    tab.fixed_section.add_files([tmp_path / "room.mp4"])
    tab.glasses_section.table.selectRow(0)

    tab.fixed_section.table.selectRow(0)

    # Only one section has a selection, so Remove is never ambiguous.
    selected = [data for section in tab.sections for data in section.selected_inputs()]
    assert selected == [tab.experiment.fixed_videos[0]]
    assert not tab.glasses_section.remove_button.isEnabled()
    assert tab.fixed_section.remove_button.isEnabled()


def test_removing_selected_inputs_removes_them(tab, changes, tmp_path):
    tab.glasses_section.add_files(
        [_glasses_video(tmp_path, "cam1"), _glasses_video(tmp_path, "cam2")]
    )
    tab.glasses_section.table.selectRow(0)
    changes.clear()

    tab.glasses_section.remove_button.click()

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam2"]
    assert tab.glasses_section.table.rowCount() == 1
    assert changes == [True]


def test_a_glasses_video_used_by_audio_cannot_be_removed(tab, tmp_path, monkeypatch):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    tab.audio_section.add_files([tmp_path / "mic1.wav"])
    tab.audio_section.table.cellWidget(0, _GLASSES).setCurrentIndex(1)
    shown = []
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.QMessageBox.critical",
        lambda *args, **kwargs: shown.append(args[2]),
    )

    tab.glasses_section.table.selectRow(0)
    tab.glasses_section.remove_button.click()

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam1"]
    assert "still used by audio inputs" in shown[0]


def test_the_sections_are_rebuilt_from_the_experiment(tab, experiment, tmp_path):
    # Another tab -- or File > Open -- changed the experiment behind this one.
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    experiment.remove_input(experiment.glasses_videos[0])

    tab.refresh()

    assert tab.glasses_section.table.rowCount() == 0


def test_set_experiment_shows_the_new_ones_inputs(tab, data_dir):
    other = Experiment(ExperimentConfig())
    InputFilesTab(other).glasses_section.add_files([data_dir / "three-people.mp4"])

    tab.set_experiment(other)

    assert tab.experiment is other
    assert all(section.experiment is other for section in tab.sections)
    assert _cells(tab.glasses_section, 0)[0] == "three-people"


def test_choosing_files_adds_them_to_that_section(tab, monkeypatch, tmp_path):
    chosen = [str(tmp_path / "room.mp4")]
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: (chosen, ""),
    )

    tab.fixed_section.add_button.click()

    assert [v.video_path for v in tab.experiment.fixed_videos] == [Path(chosen[0])]
    assert tab.experiment.glasses_videos == []


def test_cancelling_the_file_dialog_adds_nothing(tab, changes, monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([], ""),
    )

    tab.audio_section.add_button.click()

    assert tab.experiment.inputs == []
    assert changes == []


def test_a_glasses_recording_folder_is_added_whole(
    tab, changes, tmp_path, monkeypatch, glasses3_recording
):
    recording = glasses3_recording(tmp_path / "1401" / "20220728T161118Z")
    _answer_folder_dialog(monkeypatch, tmp_path / "1401")

    tab.glasses_section.add_folder_button.click()

    added = tab.experiment.glasses_videos
    assert len(added) == 1
    # Named for the participant the glasses recorded, not for the folder.
    assert added[0].id == "1401"
    assert added[0].video_path == recording / "scenevideo.mp4"
    assert added[0].gaze_path == recording
    assert changes == [True]


def test_a_glasses_2_recording_folder_is_added_whole(
    tab, tmp_path, monkeypatch, glasses2_recording
):
    recording = glasses2_recording(tmp_path / "1403" / "es6hcql")
    _answer_folder_dialog(monkeypatch, recording)

    tab.glasses_section.add_folder_button.click()

    added = tab.experiment.glasses_videos
    assert len(added) == 1
    assert added[0].id == "1403"
    assert added[0].video_path == recording / "segments" / "1" / "fullstream.mp4"
    assert added[0].gaze_path == recording


def test_a_folder_of_recordings_adds_every_one_of_them(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    card = tmp_path / "card"
    for name in ("20220728T161118Z", "20220728T170000Z"):
        glasses3_recording(card / name)
    _answer_folder_dialog(monkeypatch, card)

    tab.glasses_section.add_folder_button.click()

    # Both recordings name the same participant, so the second id is made unique.
    assert [video.id for video in tab.experiment.glasses_videos] == ["1401", "1401-2"]
    assert messages == ["Added 2 glasses recordings"]


def test_a_recording_without_its_video_is_left_out(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    """A folder whose scene video never made it off the card cannot be an input."""
    recording = glasses3_recording(tmp_path / "1401" / "20220728T161118Z")
    (recording / "scenevideo.mp4").unlink()
    _answer_folder_dialog(monkeypatch, recording)

    tab.glasses_section.add_folder_button.click()

    assert tab.experiment.glasses_videos == []
    assert "no video to go with it" in messages[0]


@pytest.mark.parametrize("generation", ["glasses2_recording", "glasses3_recording"])
def test_a_video_inside_a_recording_folder_takes_the_folder_as_its_gaze(
    tab, tmp_path, monkeypatch, request, generation
):
    """Picking the scene video out of its folder needs no second question."""
    recording = request.getfixturevalue(generation)(tmp_path / "20220728T161118Z")
    asked = _answer_gaze_dialog(monkeypatch, "")

    video = next(recording.rglob("*.mp4"))
    tab.glasses_section.add_files([video])

    assert tab.experiment.glasses_videos[0].gaze_path == recording
    assert asked == []


@pytest.mark.parametrize(
    ("by_video", "failure"),
    [
        (False, "segments"),
        (True, "segments"),
        (False, "metadata"),
        (True, "metadata"),
        (False, "missing_segments"),
    ],
)
def test_rejected_recordings_do_not_abort_successful_imports(
    tab,
    changes,
    messages,
    tmp_path,
    monkeypatch,
    glasses2_recording,
    by_video,
    failure,
):
    card = tmp_path / "card"
    first = glasses2_recording(card / "1")
    rejected = glasses2_recording(card / "2")
    last = glasses2_recording(card / "3")
    if failure == "segments":
        (rejected / "segments" / "2").mkdir()
    elif failure == "metadata":
        (rejected / "participant.json").write_text("invalid JSON")
    else:
        (rejected / "segments" / "1").rename(rejected / "segments" / "invalid")
    asked = _answer_gaze_dialog(monkeypatch, "")

    if by_video:
        tab.glasses_section.add_files(
            [
                folder / "segments" / "1" / "fullstream.mp4"
                for folder in (first, rejected, last)
            ]
        )
    else:
        _answer_folder_dialog(monkeypatch, card)
        tab.glasses_section.add_folder_button.click()

    assert [video.gaze_path for video in tab.experiment.glasses_videos] == [first, last]
    assert tab.glasses_section.table.rowCount() == 2
    assert changes == [True]
    assert any(
        str(rejected) in message and "not added" in message for message in messages
    )
    assert asked == []


@pytest.mark.parametrize("by_video", [False, True])
@pytest.mark.parametrize(
    ("participant", "expected"),
    [
        ("Subject [01]", "Subject _01_"),
        ("A/B", "A_B"),
        ("A\\B", "A_B"),
        (".", "input"),
        ("..", "input"),
        ("   ", "input"),
    ],
)
def test_participant_names_make_valid_unique_input_ids(
    tab,
    tmp_path,
    monkeypatch,
    glasses2_recording,
    participant,
    expected,
    by_video,
):
    folders = [glasses2_recording(tmp_path / name) for name in ("one", "two")]
    for folder in folders:
        (folder / "participant.json").write_text(
            json.dumps({"pa_info": {"Name": participant}})
        )

    if by_video:
        _answer_gaze_dialog(monkeypatch, "")
        tab.glasses_section.add_files(
            [folder / "segments" / "1" / "fullstream.mp4" for folder in folders]
        )
    else:
        _answer_folder_dialog(monkeypatch, *folders)
        tab.glasses_section.add_folder_button.click()

    assert [video.id for video in tab.experiment.glasses_videos] == [
        expected,
        f"{expected}-2",
    ]


def test_the_gaze_source_can_be_changed_to_a_recording_folder(
    tab, changes, tmp_path, monkeypatch, glasses3_recording
):
    recording = glasses3_recording(tmp_path / "20220728T161118Z")
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    _answer_gaze_folder_dialog(monkeypatch, recording)
    changes.clear()

    _gaze_action(tab.glasses_section, 0, GAZE_FOLDER_ACTION).trigger()

    assert tab.experiment.glasses_videos[0].gaze_path == recording
    # A folder reads as one in the table, so it is not taken for a file.
    assert tab.glasses_section.table.cellWidget(0, _GAZE).text() == (
        "20220728T161118Z/"
    )
    assert changes == [True]


def test_changing_to_a_folder_recorded_with_another_video_warns(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    """The videos people were given are often cut from the ones on the card."""
    recording = glasses3_recording(tmp_path / "20220728T161118Z")
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    _answer_gaze_folder_dialog(monkeypatch, recording)

    _gaze_action(tab.glasses_section, 0, GAZE_FOLDER_ACTION).trigger()

    assert "was recorded with scenevideo.mp4, not cam1.mp4" in messages[-1]


def test_a_folder_that_is_not_a_recording_is_refused_as_a_gaze_source(
    tab, changes, messages, tmp_path, monkeypatch
):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _answer_gaze_folder_dialog(monkeypatch, elsewhere)
    changes.clear()

    _gaze_action(tab.glasses_section, 0, GAZE_FOLDER_ACTION).trigger()

    assert tab.experiment.glasses_videos[0].gaze_path == tmp_path / "cam1.tsv"
    assert changes == []
    assert "elsewhere is not a glasses recording folder" in messages[-1]


def test_a_video_from_a_recording_folder_is_named_the_same_either_way(
    tab, tmp_path, monkeypatch, glasses3_recording
):
    """The same recording gets the same id however the user picked it."""
    by_folder = glasses3_recording(tmp_path / "one" / "20220728T161118Z")
    by_video = glasses3_recording(tmp_path / "two" / "20220728T161118Z")
    _answer_folder_dialog(monkeypatch, by_folder)

    tab.glasses_section.add_folder_button.click()
    tab.glasses_section.add_files([by_video / "scenevideo.mp4"])

    # Not "scenevideo", which is what every Glasses 3 recording calls its video.
    assert [video.id for video in tab.experiment.glasses_videos] == ["1401", "1401-2"]


def test_a_video_with_a_gaze_file_beside_it_is_still_named_for_the_video(tab, tmp_path):
    tab.glasses_section.add_files([_glasses_video(tmp_path, "cam1")])

    assert tab.experiment.glasses_videos[0].id == "cam1"


def test_the_same_video_cannot_be_added_twice(tab, changes, messages, tmp_path):
    video = _glasses_video(tmp_path, "cam1")
    tab.glasses_section.add_files([video])
    changes.clear()

    tab.glasses_section.add_files([video])

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam1"]
    assert changes == []
    assert "cam1.mp4 is already an input, as 'cam1'" in messages[-1]


def test_a_video_already_added_is_not_asked_about_again(tab, tmp_path, monkeypatch):
    """A file that will be refused should not first put up a dialog."""
    video = tmp_path / "cam1.mp4"  # no gaze file beside it, so one is asked for
    other = _write_gaze(tmp_path / "gaze.tsv")
    _answer_gaze_dialog(monkeypatch, other)
    tab.glasses_section.add_files([video])
    asked = _answer_gaze_dialog(monkeypatch, other)

    tab.glasses_section.add_files([video])

    assert asked == []


def test_the_same_video_cannot_be_added_to_two_different_sections(
    tab, messages, tmp_path
):
    video = _glasses_video(tmp_path, "cam1")
    tab.glasses_section.add_files([video])

    tab.fixed_section.add_files([video])

    assert tab.experiment.fixed_videos == []
    assert "cam1.mp4 is already an input, as 'cam1'" in messages[-1]


def test_the_same_recording_folder_cannot_be_added_twice(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    recording = glasses3_recording(tmp_path / "20220728T161118Z")
    _answer_folder_dialog(monkeypatch, recording)
    tab.glasses_section.add_folder_button.click()

    tab.glasses_section.add_folder_button.click()

    assert [v.id for v in tab.experiment.glasses_videos] == ["1401"]
    assert "scenevideo.mp4 is already an input, as '1401'" in messages[-1]


def test_a_folder_and_the_video_inside_it_are_the_same_recording(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    """Adding a recording, then its video, is adding it twice by another route."""
    recording = glasses3_recording(tmp_path / "20220728T161118Z")
    _answer_folder_dialog(monkeypatch, recording)
    tab.glasses_section.add_folder_button.click()

    tab.glasses_section.add_files([recording / "scenevideo.mp4"])

    assert [v.id for v in tab.experiment.glasses_videos] == ["1401"]
    assert "scenevideo.mp4 is already an input, as '1401'" in messages[-1]


def test_a_card_holding_a_recording_already_added_adds_only_the_rest(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    card = tmp_path / "card"
    first = glasses3_recording(card / "20220728T161118Z")
    glasses3_recording(card / "20220728T170000Z")
    _answer_folder_dialog(monkeypatch, first)
    tab.glasses_section.add_folder_button.click()
    _answer_folder_dialog(monkeypatch, card)

    tab.glasses_section.add_folder_button.click()

    assert len(tab.experiment.glasses_videos) == 2
    assert messages[-1] == "Added 1 glasses recording"


def test_several_recording_folders_can_be_picked_at_once(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    first = glasses3_recording(tmp_path / "one" / "20220728T161118Z")
    second = glasses3_recording(tmp_path / "two" / "20220728T170000Z")
    _answer_folder_dialog(monkeypatch, first, second)

    tab.glasses_section.add_folder_button.click()

    assert len(tab.experiment.glasses_videos) == 2
    assert messages[-1] == "Added 2 glasses recordings"


def test_a_folder_picked_alongside_one_holding_it_is_not_added_twice(
    tab, messages, tmp_path, monkeypatch, glasses3_recording
):
    card = tmp_path / "card"
    recording = glasses3_recording(card / "20220728T161118Z")
    _answer_folder_dialog(monkeypatch, card, recording)

    tab.glasses_section.add_folder_button.click()

    assert len(tab.experiment.glasses_videos) == 1
    assert messages[-1] == "Added 1 glasses recording"


def test_recordings_filed_deeper_than_one_folder_down_are_still_found(
    tab,
    tmp_path,
    monkeypatch,
    glasses3_recording,
    glasses2_recording,
):
    """As they are on a card: Glasses 2 keeps them three folders down."""
    session = tmp_path / "session"
    glasses3_recording(session / "1401" / "20220728T161118Z")
    glasses2_recording(session / "projects" / "sr2ixeu" / "recordings" / "es6hcql")
    _answer_folder_dialog(monkeypatch, session)

    tab.glasses_section.add_folder_button.click()

    assert sorted(v.id for v in tab.experiment.glasses_videos) == ["1401", "1403"]


def test_a_folder_holding_nothing_says_what_would_have_worked(
    tab, messages, tmp_path, monkeypatch
):
    empty = tmp_path / "holiday photos"
    (empty / "nested").mkdir(parents=True)
    _answer_folder_dialog(monkeypatch, empty)

    tab.glasses_section.add_folder_button.click()

    assert tab.experiment.glasses_videos == []
    assert messages[-1] == (
        "No glasses recording in holiday photos: open a recording folder, or "
        "one with recordings somewhere inside it"
    )


def test_the_folder_chooser_is_built_to_take_more_than_one(qtbot, monkeypatch):
    """Qt's ready-made chooser takes one folder, so this one is built to."""
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QFileDialog,
        QListView,
        QTreeView,
    )

    from body_eye_sync.gui.tabs.input_files import choose_folders

    seen = {}

    def exec_(dialog):
        seen["mode"] = dialog.fileMode()
        seen["views"] = [
            view.selectionMode()
            for view in [
                *dialog.findChildren(QListView),
                *dialog.findChildren(QTreeView),
            ]
        ]
        return 0  # the user cancelled

    monkeypatch.setattr(QFileDialog, "exec", exec_)
    widget = QWidget()
    qtbot.addWidget(widget)

    assert choose_folders(widget, "Add folders") == []
    assert seen["mode"] == QFileDialog.FileMode.Directory
    assert seen["views"]
    assert all(
        mode == QAbstractItemView.SelectionMode.ExtendedSelection
        for mode in seen["views"]
    )
