from pathlib import Path

from body_eye_sync import cli
from body_eye_sync.cli import main
from body_eye_sync.experiment.config import ExperimentConfig, GlassesVideoInput
from body_eye_sync.experiment.experiment import Experiment

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
