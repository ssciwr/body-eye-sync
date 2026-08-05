from pathlib import Path

import pytest

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import ExperimentConfig
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo
from body_eye_sync.gui.tabs.input_files import InputFilesTab

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
    assert actions.itemAt(1).widget() is section.add_button


def test_add_glasses_video_uses_the_filename_as_its_id(tab, changes, data_dir):
    video = data_dir / "three-people.mp4"

    tab.add_glasses_videos([video])

    assert [v.id for v in tab.experiment.glasses_videos] == ["three-people"]
    assert tab.experiment.glasses_videos[0].video_path == video
    assert _cells(tab.glasses_section, 0) == ["three-people", str(video)]
    assert tab.glasses_section.table.isVisibleTo(tab.glasses_section)
    assert not tab.glasses_section.empty_label.isVisibleTo(tab.glasses_section)
    assert changes == [True]


def test_each_input_goes_in_its_own_section(tab, data_dir, tmp_path):
    tab.add_audio([tmp_path / "mic1.wav"])
    tab.add_fixed_videos([tmp_path / "room.mp4"])
    tab.add_glasses_videos([data_dir / "three-people.mp4"])

    assert [type(i) for i in tab.experiment.inputs] == [GlassesVideo, FixedVideo, Audio]
    assert [section.table.rowCount() for section in tab.sections] == [1, 1, 1]
    assert _cells(tab.fixed_section, 0)[0] == "room"
    assert _cells(tab.audio_section, 0)[0] == "mic1"


def test_each_type_has_the_extra_columns_it_needs(tab, tmp_path):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    tab.add_fixed_videos([tmp_path / "room.mp4"])
    tab.add_audio([tmp_path / "mic1.wav"])

    # An extra column, and its widget, is only where it means something.
    assert _headers(tab.glasses_section) == ["Id", "File", "Gaze file"]
    assert _headers(tab.fixed_section) == ["Id", "File"]
    assert _headers(tab.audio_section) == ["Id", "File", "Glasses video"]
    assert tab.glasses_section.table.cellWidget(0, _GAZE) is not None
    assert tab.fixed_section.table.cellWidget(0, _EXTRA) is None
    assert tab.audio_section.table.cellWidget(0, _GLASSES) is not None


def test_a_section_is_only_as_tall_as_its_rows(tab, tmp_path):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    one_row = tab.glasses_section.table.height()

    tab.add_glasses_videos([_glasses_video(tmp_path, "cam2")])

    assert tab.glasses_section.table.height() > one_row


def test_adding_the_same_filename_twice_gives_unique_ids(tab, data_dir, tmp_path):
    other = tmp_path / "three-people.mp4"
    other.write_bytes(b"")

    tab.add_glasses_videos([data_dir / "three-people.mp4"])
    tab.add_fixed_videos([other])

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

    tab.add_glasses_videos([data_dir / "three-people.mp4"])

    assert tab.experiment.glasses_videos[0].gaze_path == data_dir / "three-people.tsv"
    assert asked == []
    assert tab.glasses_section.table.cellWidget(0, _GAZE).text() == "three-people.tsv"


def test_a_glasses_video_without_one_beside_it_asks(tab, tmp_path, monkeypatch):
    gaze = tmp_path / "elsewhere" / "gaze.tsv"
    gaze.parent.mkdir()
    _write_gaze(gaze)
    asked = _answer_gaze_dialog(monkeypatch, gaze)

    tab.add_glasses_videos([tmp_path / "cam1.mp4"])

    assert asked == ["Gaze file for cam1.mp4"]
    assert tab.experiment.glasses_videos[0].gaze_path == gaze


def test_a_glasses_video_is_not_added_without_a_gaze_file(
    tab, changes, messages, tmp_path, monkeypatch
):
    _answer_gaze_dialog(monkeypatch, "")  # the user cancelled the dialog

    tab.add_glasses_videos([tmp_path / "cam1.mp4"])

    assert tab.experiment.glasses_videos == []
    assert messages == ["cam1.mp4 not added: it needs a gaze file"]
    assert changes == []


def test_the_gaze_file_can_be_changed(tab, changes, tmp_path, monkeypatch):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    other = _write_gaze(tmp_path / "corrected.tsv")
    _answer_gaze_dialog(monkeypatch, other)
    changes.clear()

    tab.glasses_section.table.cellWidget(0, _GAZE).click()

    assert tab.experiment.glasses_videos[0].gaze_path == other
    assert tab.glasses_section.table.cellWidget(0, _GAZE).text() == "corrected.tsv"
    assert changes == [True]


def test_a_missing_gaze_file_is_flagged(tab, tmp_path, monkeypatch):
    _answer_gaze_dialog(monkeypatch, tmp_path / "never-exported.tsv")

    tab.add_glasses_videos([tmp_path / "cam1.mp4"])

    button = tab.glasses_section.table.cellWidget(0, _GAZE)
    assert button.text() == "never-exported.tsv (not found)"


def test_a_missing_file_is_flagged(tab, tmp_path):
    tab.add_glasses_videos([_glasses_video(tmp_path, "gone")])

    assert tab.glasses_section.table.item(0, _FILE).text().endswith("(not found)")


def test_renaming_a_row_renames_the_input(tab, changes, data_dir):
    tab.add_glasses_videos([data_dir / "three-people.mp4"])
    changes.clear()

    tab.glasses_section.table.item(0, _ID).setText("cam1")

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam1"]
    assert changes == [True]


@pytest.mark.parametrize("new_id", ["room", "", "sub/cam1"])
def test_an_invalid_id_is_reported_and_reverted(tab, messages, tmp_path, new_id):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    tab.add_fixed_videos([tmp_path / "room.mp4"])

    tab.glasses_section.table.item(0, _ID).setText(new_id)

    assert [i.id for i in tab.experiment.inputs] == ["cam1", "room"]
    assert tab.glasses_section.table.item(0, _ID).text() == "cam1"
    assert len(messages) == 1
    assert "Could not rename input" in messages[0]


def test_audio_can_be_pointed_at_a_glasses_video(tab, tmp_path):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    tab.add_audio([tmp_path / "mic1.wav"])
    combo = tab.audio_section.table.cellWidget(0, _GLASSES)
    assert [combo.itemText(i) for i in range(combo.count())] == ["—", "cam1"]

    combo.setCurrentIndex(1)

    assert tab.experiment.audio[0].glasses_video is tab.experiment.glasses_videos[0]


def test_a_new_glasses_video_becomes_a_choice_for_the_audio(tab, tmp_path):
    tab.add_audio([tmp_path / "mic1.wav"])

    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])

    combo = tab.audio_section.table.cellWidget(0, _GLASSES)
    assert [combo.itemText(i) for i in range(combo.count())] == ["—", "cam1"]


def test_selecting_in_one_section_clears_the_others(tab, tmp_path):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    tab.add_fixed_videos([tmp_path / "room.mp4"])
    tab.glasses_section.table.selectRow(0)

    tab.fixed_section.table.selectRow(0)

    # Only one section has a selection, so Remove is never ambiguous.
    assert tab.selected_inputs() == [tab.experiment.fixed_videos[0]]
    assert not tab.glasses_section.remove_button.isEnabled()
    assert tab.fixed_section.remove_button.isEnabled()


def test_removing_selected_inputs_removes_them(tab, changes, tmp_path):
    tab.add_glasses_videos(
        [_glasses_video(tmp_path, "cam1"), _glasses_video(tmp_path, "cam2")]
    )
    tab.glasses_section.table.selectRow(0)
    changes.clear()

    tab.glasses_section.remove_button.click()

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam2"]
    assert tab.glasses_section.table.rowCount() == 1
    assert changes == [True]


def test_a_glasses_video_used_by_audio_cannot_be_removed(tab, tmp_path, monkeypatch):
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    tab.add_audio([tmp_path / "mic1.wav"])
    tab.audio_section.table.cellWidget(0, _GLASSES).setCurrentIndex(1)
    shown = []
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.input_files.QMessageBox.critical",
        lambda *args, **kwargs: shown.append(args[2]),
    )

    tab.remove_inputs([tab.experiment.glasses_videos[0]])

    assert [v.id for v in tab.experiment.glasses_videos] == ["cam1"]
    assert "still used by audio inputs" in shown[0]


def test_the_sections_are_rebuilt_from_the_experiment(tab, experiment, tmp_path):
    # Another tab -- or File > Open -- changed the experiment behind this one.
    tab.add_glasses_videos([_glasses_video(tmp_path, "cam1")])
    experiment.remove_input(experiment.glasses_videos[0])

    tab.refresh()

    assert tab.glasses_section.table.rowCount() == 0


def test_set_experiment_shows_the_new_ones_inputs(tab, data_dir):
    other = Experiment(ExperimentConfig())
    InputFilesTab(other).add_glasses_videos([data_dir / "three-people.mp4"])

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
