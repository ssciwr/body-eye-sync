import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    ExperimentConfig,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.experiment import Experiment


def _config(**overrides):
    kwargs = dict(
        name="demo",
        inputs=[VideoInput(id="cam1", path="videos/session1.mp4")],
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_save_load_round_trips_the_config(tmp_path):
    original = _config(
        object_tracking=ObjectTrackingStep(detector="yolov8m"),
        face_detection=FaceDetectionStep(det_thresh=0.7),
    )
    Experiment(original, tmp_path).save()

    loaded = Experiment.load(tmp_path)
    assert loaded.folder == tmp_path
    assert loaded.config.model_dump() == original.model_dump()


def test_save_returns_config_file_and_folder_defaults(tmp_path):
    exp = Experiment(_config())
    exp.save(tmp_path)  # first save sets the folder
    assert exp.folder == tmp_path
    assert (tmp_path / "experiment.yaml").exists()
    exp.config.name = "renamed"
    exp.save()  # subsequent save reuses the folder
    assert Experiment.load(tmp_path).config.name == "renamed"


def test_relative_paths_resolve_against_folder(tmp_path):
    exp = Experiment(_config(), tmp_path)
    resolved = exp.resolved_input_path(exp.config.inputs[0])
    assert resolved == (tmp_path / "videos" / "session1.mp4").resolve()


def test_absolute_paths_pass_through(tmp_path):
    absolute = tmp_path / "clip.mp4"
    exp = Experiment(_config(inputs=[VideoInput(id="cam1", path=absolute)]))
    assert exp.resolved_input_path(exp.config.inputs[0]) == absolute


def test_output_paths_are_under_outputs_dir(tmp_path):
    exp = Experiment(_config(), tmp_path)
    assert exp.output_dir == tmp_path / "outputs"
    assert (
        exp.output_path(exp.config.inputs[0]) == tmp_path / "outputs" / "cam1.parquet"
    )


def test_paths_require_a_folder():
    exp = Experiment(_config())  # relative input path, no folder
    with pytest.raises(ValueError, match="no folder"):
        _ = exp.output_dir
    with pytest.raises(ValueError, match="no folder"):
        exp.resolved_input_path(exp.config.inputs[0])


def test_video_is_created_once_and_cached(tmp_path):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    exp = Experiment(_config(inputs=[VideoInput(id="cam1", path=video_file)]))
    spec = exp.config.inputs[0]

    video = exp.video(spec)
    assert video.video_path == video_file
    assert exp.video(spec) is video  # same instance on re-access


def test_save_writes_and_load_rehydrates_video_results(tmp_path):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    config = ExperimentConfig(
        name="demo", inputs=[VideoInput(id="cam1", path=video_file)]
    )
    exp = Experiment(config, tmp_path)
    exp.video(config.inputs[0]).set_data(
        pd.DataFrame(
            {
                "frame": [0, 0],
                "track_id": [1, 2],
                "x1": [0.0, 5.0],
                "y1": [0.0, 5.0],
                "x2": [1.0, 6.0],
                "y2": [1.0, 6.0],
                "conf": [0.9, 0.9],
            }
        )
    )
    exp.save()
    assert (tmp_path / "outputs" / "cam1.parquet").exists()

    reloaded = Experiment.load(tmp_path)
    video = reloaded.video(reloaded.config.inputs[0])
    assert video.data is not None
    assert video.data["track_id"].nunique() == 2


def test_newer_file_version_rejected(tmp_path):
    Experiment(_config(), tmp_path).save()
    path = tmp_path / "experiment.yaml"
    path.write_text(
        path.read_text().replace(
            f"version: {CURRENT_VERSION}", f"version: {CURRENT_VERSION + 1}"
        )
    )
    with pytest.raises(ValueError, match="newer than supported"):
        Experiment.load(tmp_path)
