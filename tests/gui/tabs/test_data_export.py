from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtCore import Qt

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.video_grid import VideoGridResult
from body_eye_sync.gui.tabs.data_export import DataExportTab


@pytest.fixture
def experiment(tmp_path):
    return Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path=tmp_path / "room.mp4"),
                FixedVideoInput(id="side", path=tmp_path / "side.mp4"),
            ],
            audio=[AudioInput(id="microphone", path=tmp_path / "microphone.wav")],
        ),
        tmp_path,
    )


@pytest.fixture
def tab(qtbot, experiment):
    widget = DataExportTab(experiment)
    qtbot.addWidget(widget)
    return widget


def _item(tab, input_id):
    return next(
        tab.input_list.item(index)
        for index in range(tab.input_list.count())
        if tab.input_list.item(index).data(Qt.ItemDataRole.UserRole) == input_id
    )


def test_every_input_is_listed_and_initially_checked(tab):
    assert tab.selected_input_ids() == ["room", "side", "microphone"]
    assert [
        tab.input_list.item(index).text() for index in range(tab.input_list.count())
    ] == [
        "room (fixed video)",
        "side (fixed video)",
        "microphone (audio)",
    ]
    assert tab.export_button.isEnabled()
    assert not tab.merged_audio_checkbox.isChecked()


def test_export_requires_at_least_one_selected_video(tab):
    _item(tab, "room").setCheckState(Qt.CheckState.Unchecked)
    _item(tab, "side").setCheckState(Qt.CheckState.Unchecked)

    assert tab.selected_input_ids() == ["microphone"]
    assert not tab.export_button.isEnabled()

    _item(tab, "side").setCheckState(Qt.CheckState.Checked)

    assert tab.export_button.isEnabled()


def test_refresh_preserves_choices_and_checks_new_inputs(tab, tmp_path):
    _item(tab, "side").setCheckState(Qt.CheckState.Unchecked)
    tab.experiment.add_audio(
        AudioInput(id="second_microphone", path=tmp_path / "second.wav")
    )

    tab.refresh()

    assert tab.selected_input_ids() == ["room", "microphone", "second_microphone"]


def test_export_passes_selection_and_merged_audio_to_backend(
    qtbot, tab, tmp_path, monkeypatch
):
    output = tmp_path / "chosen.mp4"
    _item(tab, "side").setCheckState(Qt.CheckState.Unchecked)
    tab.merged_audio_checkbox.setChecked(True)
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.data_export.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output), "MP4 video (*.mp4)"),
    )
    calls = []
    video_result = VideoGridResult(
        path=output,
        experiment_start=0.0,
        experiment_end=2.0,
        columns=1,
        rows=1,
        audio_tracks=("room", "microphone", "Merged audio"),
    )

    def export(experiment, path, **kwargs):
        calls.append((experiment, path, kwargs))
        kwargs["progress"](0.5)
        kwargs["progress"](1.0)
        return video_result

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.data_export.construct_video_grid", export
    )
    busy = []
    progress = []
    messages = []
    tab.busy_changed.connect(busy.append)
    tab.progress_changed.connect(lambda *values: progress.append(values))
    tab.status_message.connect(messages.append)

    tab.export_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy())

    assert calls == [
        (
            tab.experiment,
            output,
            {
                "input_ids": ["room", "microphone"],
                "include_merged_audio": True,
                "overwrite": True,
                "progress": calls[0][2]["progress"],
            },
        )
    ]
    assert busy == [True, False]
    assert (50, 100, "Exporting combined video…") in progress
    assert progress[-1] == (100, 100, "Exporting combined video…")
    assert messages == [f"Exported combined video to {output}"]
    assert tab.result_label.text() == messages[0]


def test_save_name_gets_an_mp4_extension(qtbot, tab, tmp_path, monkeypatch):
    chosen = tmp_path / "without_extension"
    seen = []
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.data_export.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(chosen), "MP4 video (*.mp4)"),
    )
    monkeypatch.setattr(tab, "_start_export", seen.append)

    tab.export_button.click()

    assert seen == [chosen.with_suffix(".mp4")]


def _turns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn_id": [0, 1],
            "start": [0.5, 1.2],
            "end": [1.0, 1.8],
            "speaker": ["room", "side"],
            "source": ["room", "side"],
            "source_segment_id": [0, 0],
            "text": ["hallo", "wie geht es"],
        }
    )


def _run_export(qtbot, tab, tmp_path, monkeypatch, output_name="chosen.mp4"):
    """Run an export whose video step is stubbed out, and return the output path."""
    output = tmp_path / output_name
    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.data_export.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output), "MP4 video (*.mp4)"),
    )

    def export(experiment, path, **kwargs):
        Path(path).write_bytes(b"")
        return VideoGridResult(
            path=Path(path),
            experiment_start=0.0,
            experiment_end=2.0,
            columns=1,
            rows=1,
            audio_tracks=("room",),
        )

    monkeypatch.setattr(
        "body_eye_sync.gui.tabs.data_export.construct_video_grid", export
    )
    tab.export_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)
    return output


def test_annotations_are_written_beside_the_video(qtbot, tab, tmp_path, monkeypatch):
    tab.experiment.speech_turns.set_data(_turns())

    output = _run_export(qtbot, tab, tmp_path, monkeypatch)

    annotations = output.with_suffix(".eaf")
    assert annotations.exists()
    assert annotations.read_text().count("<TIER ") == 2
    assert "wrote 2 annotations for 2 speaker(s)" in tab.result_label.text()


def test_an_experiment_with_no_speech_has_nothing_to_annotate(
    qtbot, tab, tmp_path, monkeypatch
):
    output = _run_export(qtbot, tab, tmp_path, monkeypatch)

    # Nothing to write is not a failure, and not worth mentioning either.
    assert output.exists()
    assert not output.with_suffix(".eaf").exists()
    assert tab.result_label.text() == f"Exported combined video to {output}"


def test_a_failed_annotation_write_does_not_hide_the_video(
    qtbot, tab, tmp_path, monkeypatch
):
    tab.experiment.speech_turns.set_data(_turns())

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("body_eye_sync.gui.tabs.data_export.export_elan", boom)

    output = _run_export(qtbot, tab, tmp_path, monkeypatch)

    # The video was written, so that is still reported, with the rest said too.
    assert output.exists()
    assert "Exported combined video" in tab.result_label.text()
    assert "could not write speech annotations: disk full" in tab.result_label.text()
