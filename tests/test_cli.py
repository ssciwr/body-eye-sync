from pathlib import Path

import numpy as np
import pandas as pd

from body_eye_sync import cli
from body_eye_sync.cli import main
from body_eye_sync.experiment.config import (
    ExperimentConfig,
    GlassesVideoInput,
    Pipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult
from body_eye_sync.pipeline.object_tracking import BoundingBox

from click.testing import CliRunner


def _make_experiment(folder):
    config = ExperimentConfig(
        glasses_videos=[
            GlassesVideoInput(
                id="cam1",
                path=Path("videos/example.mp4"),
                gaze_path=Path("videos/example.tsv"),
            )
        ],
    )
    Experiment(config, folder).save()


def test_cli_help():
    result = CliRunner().invoke(main, ("--help",))
    assert result.exit_code == 0
    assert "FOLDER" in result.output


def test_cli_version():
    from importlib.metadata import version

    result = CliRunner().invoke(main, ("--version",))
    assert result.exit_code == 0
    assert version("body-eye-sync") in result.output


def test_missing_folder_is_rejected():
    result = CliRunner().invoke(main, ("does-not-exist",))
    assert result.exit_code != 0


def test_yaml_file_is_rejected(tmp_path):
    # Only a folder is accepted, not an experiment.yaml file (file_okay=False).
    folder = tmp_path / "exp"
    _make_experiment(folder)
    result = CliRunner().invoke(main, (str(folder / "experiment.yaml"),))
    assert result.exit_code != 0


def test_runs_experiment_and_prints_results(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    _make_experiment(folder)

    captured = {}

    def fake_run_experiment(experiment, **kwargs):
        captured["experiment"] = experiment
        captured["kwargs"] = kwargs
        return {"cam1": folder / "outputs" / "cam1" / "results.parquet"}

    monkeypatch.setattr(cli, "run_experiment", fake_run_experiment)

    result = CliRunner().invoke(main, (str(folder),))

    assert result.exit_code == 0, result.output
    assert isinstance(captured["experiment"], Experiment)
    # Device/providers/output-dir are auto: only the folder and force are passed.
    assert captured["kwargs"] == {"force": False}
    assert "cam1:" in result.output


def test_force_flag_is_forwarded(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    _make_experiment(folder)

    captured = {}
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda experiment, **kwargs: captured.update(kwargs) or {},
    )

    result = CliRunner().invoke(main, (str(folder), "--force"))
    assert result.exit_code == 0, result.output
    assert captured["force"] is True


def test_cli_clusters_cached_glasses_results_and_saves_identities(tmp_path):
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id=video_id, path=f"{video_id}.mp4", gaze_path=f"{video_id}.tsv"
                )
                for video_id in ["a", "b"]
            ],
            pipeline=Pipeline(speech=None),
        ),
        tmp_path,
    )
    for index, video in enumerate(experiment.glasses_videos):
        video.set_data(
            pd.DataFrame(
                {
                    "frame": range(30),
                    "track_id": 1,
                    "x1": 0.0,
                    "y1": 0.0,
                    "x2": 1.0,
                    "y2": 1.0,
                    "conf": 0.9,
                }
            )
        )
        video.begin_face_detection(embeddings_per_track=1)
        for frame in range(30):
            video.add_face_detection_frame(
                FaceFrameResult(
                    frame,
                    [
                        FaceBox(
                            BoundingBox(0.0, 0.0, 1.0, 1.0, 1),
                            0.9,
                            landmarks=[(0.0, 0.0)] * 5,
                            embedding=np.eye(2)[index],
                        )
                    ],
                )
            )
        video.finish_face_detection()
    experiment.save()

    result = CliRunner().invoke(main, [str(tmp_path)])

    assert result.exit_code == 0, result.output
    expected = pd.DataFrame(
        [
            ("a", 1, "b"),
            ("b", 1, "a"),
        ],
        columns=["video_id", "track_id", "participant_id"],
    )
    participant_dtype = pd.CategoricalDtype(categories=["a", "b"])
    expected = expected.astype(
        {"video_id": participant_dtype, "participant_id": participant_dtype}
    )
    pd.testing.assert_frame_equal(Experiment.load(tmp_path).identities.data, expected)
    assert (tmp_path / "outputs" / "identities.parquet").exists()
