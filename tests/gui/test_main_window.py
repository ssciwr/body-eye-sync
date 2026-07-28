import pandas as pd
import pytest
from qtpy.QtWidgets import QMessageBox

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    GlassesVideoInput,
    ObjectTrackingStep,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui import MainWindow
from body_eye_sync.gui.tabs import TAB_TYPES, InputFilesTab, VideoProcessingTab

TAB_TITLES = [
    "Input files",
    "Alignment",
    "Video processing",
    "Audio processing",
    "Post processing",
    "Data export",
]


@pytest.fixture(autouse=True)
def unsaved_changes_answer(monkeypatch):
    """Answer the unsaved-changes prompt, so no test can block on it.

    Anything that drops the open experiment asks about unsaved changes first,
    and a modal question nobody answers hangs the run. Discarding is the
    answer that changes nothing; the tests about the prompt patch this again
    with the button they are exercising.
    """
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    # qtbot closes the window after the patch above is undone, so leave nothing
    # for the prompt to trigger on.
    win._dirty = False


def _experiment_folder(folder, video, with_output):
    """A saved experiment folder for ``video``, optionally with a cached output."""
    config = ExperimentConfig(
        glasses_videos=[
            GlassesVideoInput(
                id="cam1", path=video, gaze_path=video.with_suffix(".tsv")
            )
        ]
    )
    experiment = Experiment(config, folder)
    if with_output:
        experiment.glasses_videos[0].set_data(
            pd.DataFrame(
                {
                    "frame": [0, 0, 0],
                    "track_id": [1, 2, 3],
                    "x1": [0.0, 310.0, 135.0],
                    "y1": [55.0, 40.0, 35.0],
                    "x2": [155.0, 460.0, 340.0],
                    "y2": [310.0, 310.0, 310.0],
                    "conf": [0.9, 0.9, 0.9],
                }
            )
        )
    experiment.save()
    return folder


def test_window_has_icon(window):
    assert not window.windowIcon().isNull()


def test_the_tabs_cover_the_whole_window(window):
    assert window.centralWidget() is window.tabs
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == TAB_TITLES


def test_every_tab_gets_the_same_experiment(window):
    assert [type(tab) for tab in window.tab_widgets] == list(TAB_TYPES)
    assert all(tab.experiment is window.experiment for tab in window.tab_widgets)
    assert all(tab.isEnabled() for tab in window.tab_widgets)


def test_a_new_window_starts_with_an_empty_unsaved_experiment(window):
    assert window.experiment.inputs == []
    assert window.experiment.folder is None
    assert window.windowTitle() == "body-eye-sync :: [unsaved experiment]"


def test_adding_an_input_reaches_the_other_tabs(window, data_dir):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])

    video_tab = window.tab(VideoProcessingTab)
    assert video_tab.video() is window.experiment.glasses_videos[0]
    assert video_tab.video_viewer.frame_count == 5


def test_open_experiment_hands_it_to_every_tab(window, data_dir, tmp_path):
    folder = _experiment_folder(
        tmp_path, data_dir / "three-people.mp4", with_output=True
    )

    window._load_experiment(folder)

    assert all(tab.experiment is window.experiment for tab in window.tab_widgets)
    assert window.tab(InputFilesTab).glasses_section.table.rowCount() == 1
    # The experiment's cached results came with it.
    video_tab = window.tab(VideoProcessingTab)
    assert video_tab.video().data["track_id"].nunique() == 3
    assert video_tab.video_viewer.frame_count == 5


def test_open_experiment_without_video_inputs_is_fine(window, tmp_path, data_dir):
    # Audio-only experiments have nothing for the video tab to show, which is
    # not an error: the other tabs still have their inputs.
    experiment = Experiment(ExperimentConfig(), tmp_path)
    InputFilesTab(experiment).add_audio([tmp_path / "mic1.wav"])
    experiment.save()

    window._load_experiment(tmp_path)

    assert window.tab(InputFilesTab).audio_section.table.rowCount() == 1
    assert window.tab(VideoProcessingTab).video() is None


def test_open_invalid_experiment_shows_error_and_changes_nothing(
    window, tmp_path, monkeypatch
):
    before = window.experiment
    shown = {}
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: shown.setdefault("called", True),
    )

    window._load_experiment(tmp_path)  # empty folder, no experiment.yaml

    assert shown.get("called")
    assert window.experiment is before


def _prompt_answer(monkeypatch, button):
    """Answer the unsaved-changes prompt with ``button``, collecting what it asked."""
    asked = []

    def question(*args, **kwargs):
        asked.append(args[2])
        return button

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


def test_new_experiment_starts_over(window, data_dir, monkeypatch):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    asked = _prompt_answer(monkeypatch, QMessageBox.StandardButton.Discard)

    window._new_experiment()

    # Starting over drops the added input, so it asks about it first.
    assert asked == ["This experiment has changes that have not been saved."]
    assert window.experiment.inputs == []
    assert window.tab(InputFilesTab).glasses_section.table.rowCount() == 0
    assert window.tab(VideoProcessingTab).video() is None
    assert window.windowTitle() == "body-eye-sync :: [unsaved experiment]"


def test_save_writes_the_experiment_and_its_results(window, data_dir, tmp_path):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    window.experiment.glasses_videos[0].set_data(
        pd.DataFrame({"frame": [0], "track_id": [1], "conf": [0.9]})
    )
    window.experiment.folder = tmp_path

    window._save_experiment()

    reloaded = Experiment.load(tmp_path)
    assert [v.id for v in reloaded.glasses_videos] == ["three-people"]
    assert [type(s) for s in reloaded.pipeline.glasses_video.steps] == [
        ObjectTrackingStep
    ]
    output = reloaded.output_dir_for(reloaded.glasses_videos[0])
    assert Video.from_directory(output).data["track_id"].tolist() == [1]


def _answer_save_dialogs(monkeypatch, location, name=("", True)):
    """Answer the two prompts a first save asks: where, then what to call it."""
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(location),
    )
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: name,
    )


def test_a_first_save_creates_the_folder_it_is_named(
    window, data_dir, tmp_path, monkeypatch
):
    # A folder chooser can only pick a folder that exists, so the name of the
    # one to make is asked for separately -- and made by the save itself.
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    _answer_save_dialogs(monkeypatch, tmp_path, name=("my-study", True))

    window._save_experiment()

    assert Experiment.load(tmp_path / "my-study").glasses_videos[0].id == "three-people"
    assert window.windowTitle() == "body-eye-sync :: [my-study]"


def test_a_first_save_without_a_name_uses_the_chosen_folder(
    window, data_dir, tmp_path, monkeypatch
):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    _answer_save_dialogs(monkeypatch, tmp_path, name=("  ", True))

    window._save_experiment()

    assert window.experiment.folder == tmp_path
    assert (tmp_path / "experiment.yaml").exists()


@pytest.mark.parametrize("answers", [{"location": ""}, {"name": ("my-study", False)}])
def test_backing_out_of_either_save_prompt_saves_nothing(
    window, tmp_path, monkeypatch, answers
):
    _answer_save_dialogs(
        monkeypatch, answers.get("location", tmp_path), answers.get("name", ("", True))
    )

    window._save_experiment()

    assert window.experiment.folder is None
    assert list(tmp_path.iterdir()) == []


def test_a_save_that_fails_is_reported_rather_than_raised(
    window, tmp_path, monkeypatch
):
    window.experiment.folder = tmp_path
    shown = []
    monkeypatch.setattr(
        "body_eye_sync.gui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: shown.append(args[2]),
    )

    def failing_save(*args, **kwargs):
        raise OSError("no")

    monkeypatch.setattr(Experiment, "save", failing_save)

    window._save_experiment()

    assert shown == ["no"]


def test_save_then_open_round_trips_through_the_window(window, data_dir, tmp_path):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    window.tab(VideoProcessingTab).pipeline_editor._sections[1].setChecked(True)
    window.experiment.folder = tmp_path
    window._save_experiment()

    window._new_experiment()
    window._load_experiment(tmp_path)

    assert [v.id for v in window.experiment.glasses_videos] == ["three-people"]
    assert len(window.experiment.pipeline.glasses_video.steps) == 2
    assert window.experiment.folder == tmp_path
    # The reloaded pipeline is what the video tab's editor shows.
    assert len(window.tab(VideoProcessingTab).pipeline_editor.enabled_steps()) == 2


def test_title_shows_the_open_experiment_folder(window, data_dir, tmp_path):
    folder = tmp_path / "boop"
    _experiment_folder(folder, data_dir / "three-people.mp4", with_output=True)

    # Nothing saved yet, so the title says so rather than naming a folder.
    assert window.windowTitle() == "body-eye-sync :: [unsaved experiment]"

    window.load_experiment(folder)
    assert window.windowTitle() == "body-eye-sync :: [boop]"

    window._new_experiment()
    assert window.windowTitle() == "body-eye-sync :: [unsaved experiment]"


def test_load_experiment_public_entry_point(window, data_dir, tmp_path):
    folder = _experiment_folder(
        tmp_path, data_dir / "three-people.mp4", with_output=True
    )

    window.load_experiment(str(folder))

    assert window.tab(VideoProcessingTab).video().data["track_id"].nunique() == 3
    assert window.experiment.folder == tmp_path


def test_a_busy_tab_locks_the_file_actions(window, data_dir, tmp_path):
    folder = _experiment_folder(
        tmp_path, data_dir / "three-people.mp4", with_output=False
    )
    window.load_experiment(folder)
    before = window.experiment

    window.tab(VideoProcessingTab).busy_changed.emit(True)

    assert not window.new_action.isEnabled()
    assert not window.open_action.isEnabled()
    assert not window.save_action.isEnabled()
    # And the actions do nothing if triggered anyway (e.g. by shortcut).
    window._new_experiment()
    assert window.experiment is before

    window.tab(VideoProcessingTab).busy_changed.emit(False)
    assert window.new_action.isEnabled()


def test_a_busy_tab_locks_the_other_tabs(window):
    running = window.tab(VideoProcessingTab)

    running.busy_changed.emit(True)

    # Only the tab running the job stays usable, so nothing else can rename or
    # remove the input it is writing into while it works.
    enabled = [
        tab for i, tab in enumerate(window.tab_widgets) if window.tabs.isTabEnabled(i)
    ]
    assert enabled == [running]

    running.busy_changed.emit(False)
    assert all(window.tabs.isTabEnabled(i) for i in range(window.tabs.count()))


def test_closing_with_unsaved_changes_can_save_them_first(
    window, data_dir, tmp_path, monkeypatch
):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    window.experiment.folder = tmp_path
    asked = _prompt_answer(monkeypatch, QMessageBox.StandardButton.Save)

    assert window.close()

    assert asked  # the prompt was shown
    assert Experiment.load(tmp_path).glasses_videos[0].id == "three-people"


def test_closing_can_discard_unsaved_changes(window, data_dir, tmp_path, monkeypatch):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    window.experiment.folder = tmp_path
    _prompt_answer(monkeypatch, QMessageBox.StandardButton.Discard)

    assert window.close()

    assert list(tmp_path.iterdir()) == []  # nothing was written


def test_backing_out_of_the_prompt_keeps_the_window_open(window, data_dir, monkeypatch):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    _prompt_answer(monkeypatch, QMessageBox.StandardButton.Cancel)

    assert not window.close()

    assert window.experiment.inputs != []


def test_closing_a_saved_experiment_asks_nothing(
    window, data_dir, tmp_path, monkeypatch
):
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    window.experiment.folder = tmp_path
    window._save_experiment()
    asked = _prompt_answer(monkeypatch, QMessageBox.StandardButton.Cancel)

    assert window.close()

    assert asked == []


def test_a_run_and_a_pipeline_edit_count_as_unsaved_changes(window, data_dir):
    # Not just the inputs: the pipeline settings and computed results are the
    # experiment too, and are lost just as easily.
    window.tab(InputFilesTab).add_glasses_videos([data_dir / "three-people.mp4"])
    window._dirty = False

    window.tab(VideoProcessingTab).pipeline_editor.changed.emit()

    assert window._dirty


def test_a_status_message_from_a_tab_reaches_the_status_bar(window):
    window.tab(InputFilesTab).status_message.emit("hello")

    assert window.statusBar().currentMessage() == "hello"
